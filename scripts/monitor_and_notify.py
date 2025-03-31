#!/usr/bin/env python
"""
Monitor vehicle detection records and send the best image to Telegram.
This script periodically checks the detection database for completed vehicle records and
sends the image with highest confidence to a configured Telegram channel.
"""
import os
import time
import logging
import telegram
import sys
import argparse
import sqlite3
from safewheels.utils.config import load_config

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("monitor_notify")


class VehicleMonitor:
    """
    Monitors vehicle detections and sends notifications with the best image
    for each completed vehicle detection sequence.
    """

    def __init__(self, config_path):
        """
        Initialize the monitor.

        Args:
            config_path: Path to the configuration file
        """
        # Load configuration
        self.config = load_config(config_path)
        self.storage_path = self.config.get("storage_path")
        self.vehicle_id_threshold_sec = self.config.get("vehicle_id_threshold_sec")

        # Get database and image directory from config
        self.db_filename = self.config.get("db_filename")
        self.images_dirname = self.config.get("images_dirname")

        # Database file path
        self.db_path = os.path.join(self.storage_path, self.db_filename)
        self.images_dir = os.path.join(self.storage_path, self.images_dirname)

        # Check if database exists
        if not os.path.exists(self.db_path):
            logger.error(f"Database file {self.db_path} doesn't exist")
            raise FileNotFoundError(f"Database file {self.db_path} not found")

        # Initialize Telegram bot from config
        self.telegram_token = self.config.get("telegram_token", "")
        self.authorized_users = self.config.get("authorized_users", [])

        if not self.telegram_token:
            logger.error("Telegram token not configured!")
            raise ValueError("Telegram token missing in config file")

        if not self.authorized_users:
            logger.error("No authorized Telegram users configured!")
            raise ValueError("No authorized users specified in config file")

        self.bot = telegram.Bot(token=self.telegram_token)

        logger.info(f"Bot will send notifications to {len(self.authorized_users)} authorized users")

        # Tracking variables
        self.last_processed_timestamp = 0

        logger.info(f"Vehicle monitor initialized with database: {self.db_path}")
        logger.info(f"Images directory: {self.images_dir}")
        logger.info(f"Check interval: {self.check_interval_sec} seconds")
        logger.info(f"Vehicle threshold: {self.vehicle_id_threshold_sec} seconds")

    def get_unprocessed_completed_vehicles(self):
        """
        Get vehicles that:
        1. Have not been processed yet (timestamp > last_processed_timestamp)
        2. Are considered completed (last detected more than vehicle_id_threshold_sec ago)
        3. With their best image (highest OCR confidence, or plate confidence, or vehicle confidence)

        Returns:
            List of vehicle records with their best image ordered by timestamp
        """
        try:
            current_time = time.time()
            conn = sqlite3.connect(self.db_path)
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()

            # Get the best image for each vehicle in a single query
            query = """
            WITH
            -- First get the latest timestamp for each vehicle
            latest_timestamps AS (
                SELECT
                    vehicle_id,
                    MAX(timestamp) as latest_timestamp
                FROM detections
                WHERE timestamp > ?
                GROUP BY vehicle_id
            ),
            -- Then filter for vehicles that haven't been detected for at least vehicle_id_threshold_sec
            completed_vehicles AS (
                SELECT
                    vehicle_id,
                    latest_timestamp
                FROM latest_timestamps
                WHERE (? - latest_timestamp) >= ?
            ),
            -- Rank images for each vehicle by confidence
            ranked_images AS (
                SELECT
                    d.*,
                    ROW_NUMBER() OVER (
                        PARTITION BY d.vehicle_id
                        ORDER BY
                            -- First priority: OCR confidence (if available)
                            CASE WHEN d.ocr_confidence > 0 THEN 1 ELSE 0 END DESC,
                            d.ocr_confidence DESC,
                            -- Second priority: Plate detection confidence
                            CASE WHEN d.plate_detection_confidence > 0 THEN 1 ELSE 0 END DESC,
                            d.plate_detection_confidence DESC,
                            -- Third priority: Vehicle confidence
                            d.vehicle_confidence DESC
                    ) as rank
                FROM detections d
                JOIN completed_vehicles cv ON d.vehicle_id = cv.vehicle_id
            )
            -- Select only the best image (highest confidence) for each vehicle
            SELECT *
            FROM ranked_images
            WHERE rank = 1
            ORDER BY timestamp ASC
            """

            cursor.execute(query, (
                self.last_processed_timestamp,
                current_time,
                self.vehicle_id_threshold_sec
            ))

            result = [dict(row) for row in cursor.fetchall()]
            conn.close()
            return result

        except sqlite3.Error as e:
            logger.error(f"Error querying database: {e}")
            return []

    def send_notification(self, record):
        """
        Send a notification with the best image for a vehicle to all authorized users.

        Args:
            record: The record with the best image

        Returns:
            True if notification was sent successfully to at least one user, False otherwise
        """
        # Get image path
        image_filename = record.get("image", "")
        if not image_filename:
            logger.error("No image filename in record")
            return False

        image_path = os.path.join(self.images_dir, image_filename)
        if not os.path.exists(image_path):
            logger.error(f"Image {image_path} does not exist")
            return False

        # Prepare caption with vehicle details
        plate_number = record.get("plate_number", "")
        vehicle_id = record.get("vehicle_id", "")
        timestamp = record.get("datetime_ms", "")

        if plate_number:
            caption = f"🚗 Vehicle detected: {vehicle_id}\n"
            caption += f"📝 License plate: {plate_number}\n"
            caption += f"🔢 OCR confidence: {float(record.get('ocr_confidence', 0)):.2f}\n"
        else:
            caption = f"🚗 Vehicle detected: {vehicle_id}\n"
            caption += "⚠️ No license plate recognized\n"

        caption += f"⏱️ Time: {timestamp}"

        # Send photo to each authorized user
        success_count = 0
        error_count = 0

        with open(image_path, 'rb') as photo_file:
            photo_bytes = photo_file.read()

            for user_id in self.authorized_users:
                try:
                    # Create InputFile from bytes to avoid reopening file for each user
                    photo = telegram.InputFile(photo_bytes, filename=f"{vehicle_id}.jpg")
                    self.bot.send_photo(chat_id=user_id, photo=photo, caption=caption)
                    success_count += 1
                except Exception as user_error:
                    logger.error(f"Failed to send notification to user {user_id}: {user_error}")
                    error_count += 1

        if success_count > 0:
            logger.info(
                f"Sent notification for vehicle {vehicle_id} to {success_count} users "
                f"({error_count} errors)"
            )
            return True
        else:
            logger.error(f"Failed to send notification to any users for vehicle {vehicle_id}")
            return False

    def process_vehicles(self):
        """
        Process unprocessed vehicles and send notifications for them.
        Update the last processed timestamp to the latest vehicle timestamp.
        """
        # Get unprocessed vehicles with their best images
        vehicles = self.get_unprocessed_completed_vehicles()

        if not vehicles:
            return

        logger.info(f"Found {len(vehicles)} unprocessed vehicles")

        # Process each vehicle
        for record in vehicles:
            vehicle_id = record["vehicle_id"]
            timestamp = float(record["timestamp"])

            logger.info(f"Processing vehicle {vehicle_id} with timestamp {timestamp}")

            # Send notification with the best image
            self.send_notification(record)

            # Update the last processed timestamp if this is the latest timestamp
            if timestamp > self.last_processed_timestamp:
                self.last_processed_timestamp = timestamp
                logger.debug(f"Updated last processed timestamp to {timestamp}")

    def run(self):
        """
        Run the monitoring loop with the configured check interval.
        """
        logger.info(f"Starting monitoring loop with interval: {self.check_interval_sec}s")

        try:
            while True:
                self.process_vehicles()
                time.sleep(self.check_interval_sec)
        except KeyboardInterrupt:
            logger.info("Monitoring stopped by user")
        except Exception as e:
            logger.error(f"Error in monitoring loop: {e}")


if __name__ == "__main__":
    # Parse command-line arguments
    parser = argparse.ArgumentParser(
        description="Monitor vehicle detection records and send notifications via Telegram"
    )
    parser.add_argument(
        "-c", "--config", required=True,
        help="Path to the configuration file"
    )
    args = parser.parse_args()

    # Verify that the config file exists
    if not os.path.exists(args.config):
        logger.error(f"Config file not found: {args.config}")
        sys.exit(1)

    # Initialize and run the monitor
    monitor = VehicleMonitor(config_path=args.config)

    # Run the monitoring loop
    monitor.run()
