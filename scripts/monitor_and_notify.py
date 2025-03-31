#!/usr/bin/env python
"""
Monitor vehicle detection records and send the best image to Telegram.
This script checks the detection CSV file for completed vehicle records and
sends the image with highest confidence to a configured Telegram channel.
"""
import os
import csv
import time
import logging
import telegram
import sys
import argparse

# Add project root to path to import safewheels modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
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
        self.vehicle_id_threshold_sec = self.config.get("vehicle_id_threshold_sec", 5)

        # Get CSV and image directory from config
        self.csv_filename = self.config.get("csv_filename")
        self.images_dirname = self.config.get("images_dirname", "images")

        # CSV file path
        self.csv_file = os.path.join(self.storage_path, self.csv_filename)
        self.images_dir = os.path.join(self.storage_path, self.images_dirname)

        # Initialize Telegram bot from config
        self.telegram_token = self.config.get("telegram_token", "")
        self.chat_id = self.config.get("telegram_chat_id", "")

        if not self.telegram_token or not self.chat_id:
            logger.error("Telegram token or chat ID not configured!")
            raise ValueError("Telegram configuration missing in config file")

        self.bot = telegram.Bot(token=self.telegram_token)

        # Tracking variables
        self.last_processed_vehicle_id = None
        self.last_processed_timestamp = 0
        self.processed_vehicle_ids = set()

        logger.info(f"Vehicle monitor initialized with CSV: {self.csv_file}")
        logger.info(f"Images directory: {self.images_dir}")
        logger.info(f"Telegram notifications will be sent to chat ID: {self.chat_id}")

    def get_latest_record(self):
        """
        Get the most recent record from the CSV file.

        Returns:
            The most recent record as a dict, or None if no records found
        """
        try:
            if not os.path.exists(self.csv_file):
                logger.warning(f"CSV file {self.csv_file} doesn't exist")
                return None

            with open(self.csv_file, 'r', newline='') as f:
                reader = csv.DictReader(f)
                records = list(reader)

            if not records:
                return None

            # Sort by timestamp (newest first)
            records.sort(key=lambda x: float(x.get("timestamp", 0)), reverse=True)
            return records[0]
        except Exception as e:
            logger.error(f"Error reading CSV file: {e}")
            return None

    def get_records_by_vehicle_id(self, vehicle_id):
        """
        Get all records for a specific vehicle ID.

        Args:
            vehicle_id: The vehicle ID to search for

        Returns:
            List of records for the specified vehicle ID
        """
        records = []
        try:
            with open(self.csv_file, 'r', newline='') as f:
                reader = csv.DictReader(f)
                records = [r for r in reader if r.get("vehicle_id") == vehicle_id]
        except Exception as e:
            logger.error(f"Error reading CSV file: {e}")
            return []

        return records

    def find_best_image_record(self, records):
        """
        Find the record with the best image based on confidence scores.

        Args:
            records: List of vehicle records

        Returns:
            The record with the highest confidence, or None if no records
        """
        if not records:
            return None

        # 1. First try to find record with highest OCR confidence (plate recognized)
        ocr_records = [r for r in records if float(r.get("ocr_confidence", 0)) > 0]
        if ocr_records:
            return max(ocr_records, key=lambda x: float(x.get("ocr_confidence", 0)))

        # 2. If no OCR confidence, use plate detection confidence
        plate_records = [r for r in records if float(r.get("plate_detection_confidence", 0)) > 0]
        if plate_records:
            return max(plate_records, key=lambda x: float(x.get("plate_detection_confidence", 0)))

        # 3. Otherwise, use vehicle confidence
        return max(records, key=lambda x: float(x.get("vehicle_confidence", 0)))

    def send_notification(self, record):
        """
        Send a notification with the best image for a vehicle.

        Args:
            record: The record with the best image
        """
        try:
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

            # Send photo
            with open(image_path, 'rb') as photo:
                self.bot.send_photo(chat_id=self.chat_id, photo=photo, caption=caption)

            logger.info(f"Sent notification for vehicle {vehicle_id}")
            return True

        except Exception as e:
            logger.error(f"Error sending notification: {e}")
            return False

    def check_and_notify(self):
        """
        Check for completed vehicle records and send notifications.
        """
        # Get the latest record
        latest_record = self.get_latest_record()
        if not latest_record:
            return

        latest_timestamp = float(latest_record.get("timestamp", 0))
        latest_vehicle_id = latest_record.get("vehicle_id", "")
        current_time = time.time()

        # If this is a different vehicle than the last processed one
        if latest_vehicle_id != self.last_processed_vehicle_id:
            # Check if enough time has passed since last detection to consider it complete
            time_since_detection = current_time - latest_timestamp

            if time_since_detection >= self.vehicle_id_threshold_sec:
                # This means the vehicle detection sequence is likely complete

                # Check if we already processed this vehicle ID
                if latest_vehicle_id not in self.processed_vehicle_ids:
                    logger.info(f"Processing completed vehicle: {latest_vehicle_id}")

                    # Get all records for this vehicle
                    vehicle_records = self.get_records_by_vehicle_id(latest_vehicle_id)

                    # Find the best image
                    best_record = self.find_best_image_record(vehicle_records)

                    if best_record:
                        # Send notification with the best image
                        self.send_notification(best_record)

                    # Add to processed set
                    self.processed_vehicle_ids.add(latest_vehicle_id)

                    # Limit the size of the processed set to avoid memory issues
                    if len(self.processed_vehicle_ids) > 1000:
                        self.processed_vehicle_ids = set(list(self.processed_vehicle_ids)[-500:])

        # Update the last processed vehicle ID and timestamp
        self.last_processed_vehicle_id = latest_vehicle_id
        self.last_processed_timestamp = latest_timestamp

    def run(self):
        """
        Run the monitoring loop with 0.1s check interval.
        """
        check_interval = 0.1  # Check every 0.1 seconds
        logger.info(f"Starting monitoring loop with interval: {check_interval}s")

        try:
            while True:
                self.check_and_notify()
                time.sleep(check_interval)
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
        "-c", "--config",
        default="config/config.json",
        help="Path to the configuration file (default: config/config.json)"
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
