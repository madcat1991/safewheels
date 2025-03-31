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

    def __init__(self, storage_path="data/vehicles", max_stored_images=1000, vehicle_id_threshold_sec=5):
        """
        Initialize the record manager.

        Args:
            storage_path: Path to store detection images and metadata
            max_stored_images: Maximum number of images to keep in storage
            vehicle_id_threshold_sec: Time threshold in seconds to consider as the same vehicle
        """
        self.storage_path = storage_path
        self.max_stored_images = max_stored_images
        self.vehicle_id_threshold_sec = vehicle_id_threshold_sec

        # Vehicle tracking variables
        self.last_detection_time = 0
        self.current_vehicle_id = self._generate_vehicle_id()

        # Ensure storage directory exists
        os.makedirs(storage_path, exist_ok=True)
        os.makedirs(os.path.join(storage_path, "images"), exist_ok=True)

        # Setup CSV file
        self.csv_file = os.path.join(storage_path, "detections.csv")
        self.csv_headers = [
            "vehicle_id",
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

        logger.info(
            f"Record manager initialized with storage at {storage_path}, "
            f"vehicle ID threshold: {vehicle_id_threshold_sec}s"
        )

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

    def _generate_vehicle_id(self):
        """Generate a unique short hash ID for a vehicle."""
        return uuid.uuid4().hex[:6]  # 6 character hex ID

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
        # Check if we need to generate a new vehicle ID based on time threshold
        if timestamp - self.last_detection_time > self.vehicle_id_threshold_sec:
            self.current_vehicle_id = self._generate_vehicle_id()
            logger.debug(f"New vehicle ID generated: {self.current_vehicle_id}")

        # Update the last detection time
        self.last_detection_time = timestamp

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
            "vehicle_id": self.current_vehicle_id,
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
            logger.info(
                f"Saved detection with ID {self.current_vehicle_id}, "
                f"plate {plate_number} (OCR confidence: {ocr_confidence:.2f})")
        else:
            logger.info(
                f"Saved vehicle detection with ID {self.current_vehicle_id} "
                f"(confidence: {vehicle_confidence:.2f})"
            )

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

        try:
            image_files = [f for f in os.listdir(image_dir) if os.path.isfile(os.path.join(image_dir, f))]
        except Exception as e:
            logger.error(f"Error listing image directory: {e}")
            return

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
        records.sort(key=lambda x: float(x.get("timestamp", 0)))

        # Select records to remove (oldest ones)
        records_to_keep = records[excess:]
        records_to_remove = records[:excess]

        # Remove images associated with the oldest records
        removed_images = 0
        for record in records_to_remove:
            # Remove vehicle image
            vehicle_img = record.get("vehicle_img", "")
            if vehicle_img:
                try:
                    image_path = os.path.join(image_dir, vehicle_img)
                    if os.path.exists(image_path):
                        os.remove(image_path)
                        removed_images += 1
                except Exception as e:
                    logger.warning(f"Failed to remove image {vehicle_img}: {e}")

            # Remove plate image if exists
            plate_img = record.get("plate_img", "")
            if plate_img:
                try:
                    image_path = os.path.join(image_dir, plate_img)
                    if os.path.exists(image_path):
                        os.remove(image_path)
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
        records.sort(key=lambda x: float(x.get("timestamp", 0)), reverse=True)
        return records[:count]

    def get_records_by_vehicle_id(self, vehicle_id):
        """
        Get all records for a specific vehicle ID.

        Args:
            vehicle_id: The vehicle ID to search for

        Returns:
            List of records for the specified vehicle ID, sorted by timestamp
        """
        records = []
        try:
            with open(self.csv_file, 'r', newline='') as f:
                reader = csv.DictReader(f)
                records = [r for r in reader if r.get("vehicle_id") == vehicle_id]
        except Exception as e:
            logger.error(f"Error reading CSV file: {e}")
            return []

        # Sort by timestamp
        records.sort(key=lambda x: float(x.get("timestamp", 0)))
        return records
