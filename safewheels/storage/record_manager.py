"""
Record manager for storing vehicle and license plate detections.
"""
import os
import csv
import logging
import cv2
from datetime import datetime
import uuid

logger = logging.getLogger(__name__)


class RecordManager:
    """
    Simple storage manager for vehicle detections and license plate information.
    Saves detection data in CSV format and manages image storage.
    """

    def __init__(self, storage_path="data/vehicles", max_stored_images=1000):
        """
        Initialize the record manager.

        Args:
            storage_path: Path to store detection images and metadata
            max_stored_images: Maximum number of images to keep in storage
        """
        self.storage_path = storage_path
        self.max_stored_images = max_stored_images

        # Ensure storage directory exists
        os.makedirs(storage_path, exist_ok=True)
        os.makedirs(os.path.join(storage_path, "images"), exist_ok=True)

        # Setup CSV file
        self.csv_file = os.path.join(storage_path, "detections.csv")
        self.csv_headers = [
            "timestamp",
            "datetime_ms",
            "vehicle_img",
            "vehicle_confidence",
            "plate_img",
            "plate_detection_confidence",
            "plate_number",
            "ocr_confidence",
            "frame_number"
        ]

        # Create CSV file if it doesn't exist
        self._init_csv_file()

        logger.info(f"Record manager initialized with storage at {storage_path}")

    def _init_csv_file(self):
        """Initialize the CSV file with headers if it doesn't exist."""
        if not os.path.exists(self.csv_file):
            try:
                with open(self.csv_file, 'w', newline='') as f:
                    writer = csv.DictWriter(f, fieldnames=self.csv_headers)
                    writer.writeheader()
                logger.info(f"Created new detections CSV file at {self.csv_file}")
            except Exception as e:
                logger.error(f"Error creating CSV file: {e}")

    def add_detection(self, timestamp, vehicle_img, vehicle_confidence=0.0, plate_img=None,
                      plate_detection_confidence=0.0, plate_number=None, ocr_confidence=0.0, frame_number=None):
        """
        Add a new vehicle detection record and save associated images.

        Args:
            timestamp: Detection time
            vehicle_img: Image of the detected vehicle
            vehicle_confidence: Confidence score for the vehicle detection
            plate_img: Image of the license plate (or None if not detected)
            plate_detection_confidence: Confidence score for plate detection
            plate_number: Recognized license plate text (or None if not recognized)
            ocr_confidence: Confidence score for the OCR recognition
            frame_number: Optional frame number (for video file processing)
        """
        # Generate filename with timestamp and recognition type
        current_time = datetime.now()
        timestamp_ms = current_time.strftime("%Y%m%d_%H%M%S_%f")  # Full timestamp with microseconds
        vehicle_filename = f"{timestamp_ms}_car.jpg"
        plate_filename = f"{timestamp_ms}_plate.jpg" if plate_img is not None else None

        # Save vehicle image
        vehicle_path = os.path.join(self.storage_path, "images", vehicle_filename)
        cv2.imwrite(vehicle_path, vehicle_img)

        # Save plate image if available
        if plate_img is not None:
            plate_path = os.path.join(self.storage_path, "images", plate_filename)
            cv2.imwrite(plate_path, plate_img)

        # Create detection record
        detection = {
            "timestamp": timestamp,
            "datetime_ms": datetime.fromtimestamp(timestamp).strftime("%Y-%m-%d %H:%M:%S.%f"),
            "vehicle_img": vehicle_filename,
            "vehicle_confidence": vehicle_confidence,
            "plate_img": plate_filename,
            "plate_detection_confidence": plate_detection_confidence,
            "plate_number": plate_number if plate_number else "",
            "ocr_confidence": ocr_confidence,
            "frame_number": frame_number if frame_number is not None else ""
        }

        # Save to CSV
        self._append_to_csv(detection)

        # Log the detection
        if plate_number:
            logger.info(f"Saved detection with plate {plate_number} (OCR confidence: {ocr_confidence:.2f})")
        else:
            logger.info(f"Saved vehicle detection (confidence: {vehicle_confidence:.2f})")

        # Ensure we don't exceed max images
        self._enforce_storage_limit()

    def _append_to_csv(self, detection):
        """Append a detection record to the CSV file."""
        try:
            with open(self.csv_file, 'a', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=self.csv_headers)
                writer.writerow(detection)
        except Exception as e:
            logger.error(f"Error appending to CSV file: {e}")

    def _enforce_storage_limit(self):
        """Ensure storage doesn't exceed maximum limit by removing oldest images and records."""
        # Count image files
        image_dir = os.path.join(self.storage_path, "images")
        image_files = [f for f in os.listdir(image_dir) if os.path.isfile(os.path.join(image_dir, f))]

        if len(image_files) <= self.max_stored_images:
            return

        # Need to remove some old images
        excess = len(image_files) - self.max_stored_images

        # Read all records from CSV
        records = []
        try:
            with open(self.csv_file, 'r', newline='') as f:
                reader = csv.DictReader(f)
                records = list(reader)
        except Exception as e:
            logger.error(f"Error reading CSV file: {e}")
            return

        # Sort records by timestamp (oldest first)
        records.sort(key=lambda x: float(x["timestamp"]))

        # Select records to remove (oldest ones)
        records_to_keep = records[excess:]
        records_to_remove = records[:excess]

        # Remove images associated with the oldest records
        removed_images = 0
        for record in records_to_remove:
            # Remove vehicle image
            vehicle_img = record["vehicle_img"]
            if vehicle_img:
                try:
                    os.remove(os.path.join(image_dir, vehicle_img))
                    removed_images += 1
                except Exception as e:
                    logger.warning(f"Failed to remove image {vehicle_img}: {e}")

            # Remove plate image if exists
            plate_img = record["plate_img"]
            if plate_img:
                try:
                    os.remove(os.path.join(image_dir, plate_img))
                    removed_images += 1
                except Exception as e:
                    logger.warning(f"Failed to remove image {plate_img}: {e}")

        # Rewrite CSV with remaining records
        try:
            with open(self.csv_file, 'w', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=self.csv_headers)
                writer.writeheader()
                writer.writerows(records_to_keep)
        except Exception as e:
            logger.error(f"Error rewriting CSV file: {e}")

        logger.info(f"Storage limit enforced: removed {len(records_to_remove)} old records and {removed_images} images")

    def get_latest_records(self, count=10):
        """
        Get the most recent vehicle records.

        Args:
            count: Number of records to return

        Returns:
            List of the most recent vehicle records
        """
        records = []
        try:
            with open(self.csv_file, 'r', newline='') as f:
                reader = csv.DictReader(f)
                records = list(reader)
        except Exception as e:
            logger.error(f"Error reading CSV file: {e}")
            return []

        # Sort by timestamp (newest first)
        records.sort(key=lambda x: float(x["timestamp"]), reverse=True)
        return records[:count]
