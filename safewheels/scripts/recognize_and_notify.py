#!/usr/bin/env python
"""
Recognize license plates from vehicle detection records and send notifications.

This script retrieves all unprocessed and completed vehicle records, processes
license plate images, recognizes plate numbers, and sends notifications with
the best results to authorized Telegram users.
"""
from datetime import datetime
import os
import time
import logging
import json
import argparse
import sqlite3
import cv2
import asyncio
import telegram

from safewheels.functions import get_compute_params, add_padding_to_bbox
from safewheels.models.plate_recognizer import EUPlateRecognizer
from safewheels.utils.config import load_config

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("recognize_notify")


class PlateRecognitionProcessor:
    """
    Processes vehicle records, recognizes license plates, and sends notifications
    with the best results to authorized Telegram users.
    """

    def __init__(self, config_path):
        """
        Initialize the processor.

        Args:
            config_path: Path to the configuration file
        """
        # Load configuration
        self.config = load_config(config_path)
        self.storage_path = self.config.get("storage_path")
        self.check_interval_sec = self.config.get("check_interval_sec", 30)
        self.ocr_confidence_threshold = self.config.get("ocr_confidence_threshold", 0.3)

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

        # Initialize plate recognizer
        use_gpu, device, _ = get_compute_params(self.config)
        self.plate_recognizer = EUPlateRecognizer(use_gpu=use_gpu, device=device)

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

        # Set up timestamp persistence
        self.timestamp_filename = self.config.get("timestamp_file", "last_processed.txt")
        self.timestamp_path = os.path.join(self.storage_path, self.timestamp_filename)

        # Load the last processed timestamp from file if it exists
        self.last_processed_timestamp = self._load_timestamp()

        logger.info(f"Plate recognition processor initialized with database: {self.db_path}")
        logger.info(f"Images directory: {self.images_dir}")
        logger.info(f"Timestamp file: {self.timestamp_path}")
        logger.info(f"Last processed timestamp: {self.last_processed_timestamp}")
        logger.info(f"Check interval: {self.check_interval_sec} seconds")

    def _load_timestamp(self):
        """
        Load the last processed timestamp from file.

        Returns:
            The last processed timestamp as a float, or 0 if file doesn't exist
        """
        try:
            if os.path.exists(self.timestamp_path):
                with open(self.timestamp_path, 'r') as f:
                    timestamp_str = f.read().strip()
                    if timestamp_str:
                        return float(timestamp_str)
            return 0
        except Exception as e:
            logger.error(f"Error loading timestamp file: {e}")
            return 0

    def _save_timestamp(self, timestamp):
        """
        Save the timestamp to file.

        Args:
            timestamp: The timestamp to save
        """
        try:
            # Ensure the directory exists
            os.makedirs(os.path.dirname(self.timestamp_path), exist_ok=True)

            with open(self.timestamp_path, 'w') as f:
                f.write(str(timestamp))
            logger.debug(f"Saved timestamp {timestamp} to {self.timestamp_path}")
        except Exception as e:
            logger.error(f"Error saving timestamp file: {e}")

    def get_unprocessed_vehicles(self):
        """
        Get all vehicles that haven't been processed yet (timestamp > last_processed_timestamp)
        and are considered completed, with all their records in a single query.

        Returns:
            Dictionary of vehicle_id -> vechicle related data
        """
        best_image_record_per_vehicle = {}
        bbox_records_per_vehicle = {}
        latest_timestamp_per_vehicle = {}

        try:
            conn = sqlite3.connect(self.db_path)
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()

            # Get all completed and unprocessed vehicle records in a single query
            # A completed vehicle is one that hasn't been detected for a certain threshold period
            current_time = time.time()
            threshold_sec = self.config.get("vehicle_id_threshold_sec", 5)

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
                )
                -- Select all records for these completed vehicles
                SELECT
                    d.id,
                    d.vehicle_id,
                    d.timestamp,
                    d.image,
                    d.vehicle_confidence,
                    d.plate_bbox,
                    d.plate_confidence
                FROM detections d
                JOIN completed_vehicles cv ON d.vehicle_id = cv.vehicle_id
                ORDER BY d.vehicle_id, d.timestamp ASC
            """

            cursor.execute(query, (
                self.last_processed_timestamp,
                current_time,
                threshold_sec
            ))
            all_records = cursor.fetchall()
            conn.close()

        except sqlite3.Error as e:
            logger.error(f"Error querying database: {e}")

        for record in all_records:
            record_dict = dict(record)
            vehicle_id = record_dict.pop("vehicle_id")

            # Prepare image path
            image = record_dict.pop("image")
            record_dict["image_path"] = os.path.join(self.images_dir, image)

            # Update bbox records per vehicle
            plate_bbox = record_dict["plate_bbox"]
            if plate_bbox is not None:
                record_dict["plate_bbox"] = json.loads(plate_bbox)
                bbox_records_per_vehicle.setdefault(vehicle_id, []).append(record_dict)

            # Update best image record per vehicle
            if vehicle_id in best_image_record_per_vehicle:
                record_vehicle_confidence = record_dict["vehicle_confidence"]
                current_vehicle_confidence = best_image_record_per_vehicle[vehicle_id]["vehicle_confidence"]
                if record_vehicle_confidence > current_vehicle_confidence:
                    best_image_record_per_vehicle[vehicle_id] = record_dict
            else:
                best_image_record_per_vehicle[vehicle_id] = record_dict

            # Update latest timestamp per vehicle
            latest_timestamp = record_dict["timestamp"]
            if vehicle_id in latest_timestamp_per_vehicle:
                current_latest_timestamp = latest_timestamp_per_vehicle[vehicle_id]
                if latest_timestamp > current_latest_timestamp:
                    latest_timestamp_per_vehicle[vehicle_id] = latest_timestamp
            else:
                latest_timestamp_per_vehicle[vehicle_id] = latest_timestamp

        vehicles = {}
        for vehicle_id in latest_timestamp_per_vehicle:
            vehicles[vehicle_id] = {
                "latest_timestamp": latest_timestamp_per_vehicle[vehicle_id],
                "best_image_record": best_image_record_per_vehicle[vehicle_id],
                "bbox_records": bbox_records_per_vehicle.get(vehicle_id, [])
            }
        logger.info(f"Found {len(vehicles)} unprocessed vehicles")

        return vehicles

    def extract_plate_image(self, image_path, plate_bbox):
        """
        Load an image and extract the license plate with padding.

        Args:
            image_path: Path to the vehicle image
            plate_bbox: Bounding box coordinates for the license plate [x, y, w, h]

        Returns:
            Cropped and padded license plate image
        """
        try:
            # Load image
            img = cv2.imread(image_path)
            if img is None:
                logger.error(f"Failed to load image: {image_path}")
                return None

            # Add padding to bbox
            padded_bbox = add_padding_to_bbox(plate_bbox, img.shape, padding_percent=0.1)
            px, py, pw, ph = padded_bbox

            # Crop the plate
            plate_img = img[py:py+ph, px:px+pw]

            return plate_img

        except Exception as e:
            logger.error(f"Error extracting plate from image: {e}")
            return None

    def get_notification_data(self, vehicle_id, vehicle_data):
        """
        Process all records for a vehicle, extract and recognize license plates,
        and prepare notification data.

        Args:
            vehicle_id: ID of the vehicle
            vehicle_data: Dict with vehicle data

        Returns:
            Dict with notification data
        """
        logger.info(f"Processing vehicle {vehicle_id}")

        latest_timestamp = vehicle_data["latest_timestamp"]
        bbox_records = vehicle_data["bbox_records"]

        best_record = None
        plate_images = []
        plate_number = None
        ocr_confidence = 0

        if bbox_records:
            for record in bbox_records:
                if best_record is None or record["plate_confidence"] > best_record["plate_confidence"]:
                    best_record = record

                plate_img = self.extract_plate_image(record["image_path"], record["plate_bbox"])
                if plate_img is not None:
                    plate_images.append(plate_img)

            if plate_images:
                # Recognize plate
                plate_number, ocr_confidence = self.plate_recognizer.recognize(
                    plate_images,
                    ocr_confidence_threshold=self.ocr_confidence_threshold
                )
        else:
            best_record = vehicle_data["best_image_record"]

        return {
            "vehicle_id": vehicle_id,
            "record": best_record,
            "plate_number": plate_number,
            "ocr_confidence": ocr_confidence,
            "latest_timestamp": latest_timestamp
        }

    async def send_notification(self, notification_data):
        """
        Send a notification with vehicle detection and license plate recognition results.

        Args:
            notification_data: Dict with vehicle_id, plate_number, ocr_confidence, and image_path

        Returns:
            True if notification was sent successfully to at least one user, False otherwise
        """
        vehicle_id = notification_data["vehicle_id"]
        plate_number = notification_data["plate_number"]
        ocr_confidence = notification_data["ocr_confidence"]
        record = notification_data["record"]

        image_path = record["image_path"]
        plate_bbox = record["plate_bbox"]
        datetime_str = datetime.fromtimestamp(record["timestamp"]).strftime("%Y-%m-%d %H:%M:%S")
        # TODO add bbox and ocr_confidence on image

        if not os.path.exists(image_path):
            logger.error(f"Image {image_path} does not exist")
            return False

        # Prepare caption
        caption = f"🚗 Vehicle detected: {vehicle_id}\n"
        if plate_number:
            caption += (
                f"📝 License plate: {plate_number}\n"
                f"🔢 OCR confidence: {ocr_confidence:.2f}\n"
            )
        elif plate_bbox:
            caption += "⚠️⚠️⚠️ No license plate recognized ⚠️⚠️⚠️\n"
        else:
            caption += "🚨🚨🚨 No license plate detected 🚨🚨🚨\n"

        caption += f"⏱️ Time: {datetime_str}"

        # Send photo to each authorized user
        success_count = 0
        error_count = 0

        with open(image_path, 'rb') as photo_file:
            photo_bytes = photo_file.read()

            for user_id in self.authorized_users:
                try:
                    # Create InputFile from bytes to avoid reopening file for each user
                    photo = telegram.InputFile(photo_bytes, filename=f"{vehicle_id}.jpg")
                    # await self.bot.send_photo(chat_id=user_id, photo=photo, caption=caption)
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

    async def process_vehicles(self):
        """
        Process unprocessed vehicles, recognize license plates, and send notifications.
        Update the last processed timestamp to the latest vehicle timestamp.
        """
        # Get unprocessed vehicles with their records
        vehicles = self.get_unprocessed_vehicles()

        # Process each vehicle
        for vehicle_id, vehicle_data in vehicles.items():
            # Process vehicle's records
            notification_data = self.get_notification_data(vehicle_id, vehicle_data)

            # Send notification
            await self.send_notification(notification_data)

            # Update the last processed timestamp
            # TODO: нужно гарантировать порядок по времени
            latest_timestamp = notification_data["latest_timestamp"]
            if latest_timestamp > self.last_processed_timestamp:
                # Add a small buffer (0.001 sec) to avoid processing
                # the same vehicle twice (due to float precision)
                self.last_processed_timestamp = latest_timestamp + 0.001
                logger.info(f"Updated last processed timestamp to {self.last_processed_timestamp}")

                # Save the timestamp to file
                self._save_timestamp(self.last_processed_timestamp)

    async def run(self):
        """
        Run the monitoring loop with the configured check interval.
        """
        logger.info(f"Starting recognition & notification loop with interval: {self.check_interval_sec}s")

        try:
            while True:
                await self.process_vehicles()
                logger.info(f"Sleeping for {self.check_interval_sec}s")
                await asyncio.sleep(self.check_interval_sec)
        except KeyboardInterrupt:
            logger.info("Processing stopped by user")
            # Save timestamp on exit
            self._save_timestamp(self.last_processed_timestamp)
        except Exception as e:
            logger.error(f"Error in processing loop: {e}")
            # Try to save timestamp even if there was an error
            self._save_timestamp(self.last_processed_timestamp)


async def main():
    # Parse command-line arguments
    parser = argparse.ArgumentParser(
        description="Recognize license plates from vehicle detection records and send notifications"
    )
    parser.add_argument("-c", "--config", required=True, help="Path to the configuration file")
    args = parser.parse_args()

    # Initialize and run the processor
    processor = PlateRecognitionProcessor(config_path=args.config)

    # Run the processing loop
    await processor.run()

if __name__ == "__main__":
    # Run the async main function
    asyncio.run(main())
