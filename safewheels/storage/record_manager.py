"""
Record manager for storing vehicle and license plate detections.
"""
import os
import json
import logging
import cv2
from datetime import datetime
import uuid

logger = logging.getLogger(__name__)


class RecordManager:
    """
    Manages storage of vehicle detections and license plate information.
    Groups multiple photos of the same car to increase recognition accuracy.
    """

    def __init__(self, storage_path="data/vehicles", max_stored_images=1000, grouping_window=10):
        """
        Initialize the record manager.

        Args:
            storage_path: Path to store detection images and metadata
            max_stored_images: Maximum number of images to keep in storage
            grouping_window: Time window in seconds to group detections of the same vehicle
        """
        self.storage_path = storage_path
        self.max_stored_images = max_stored_images
        self.grouping_window = grouping_window

        # Ensure storage directory exists
        os.makedirs(storage_path, exist_ok=True)
        os.makedirs(os.path.join(storage_path, "images"), exist_ok=True)

        # Load existing records
        self.records = []
        self.records_file = os.path.join(storage_path, "records.json")
        self._load_records()

        # Active vehicle groups (currently being tracked)
        self.active_groups = {}

        logger.info(f"Record manager initialized with storage at {storage_path}")
        logger.info(f"Loaded {len(self.records)} existing records")

    def _load_records(self):
        """Load existing records from JSON file."""
        if os.path.exists(self.records_file):
            try:
                with open(self.records_file, 'r') as f:
                    self.records = json.load(f)
            except Exception as e:
                logger.error(f"Error loading records: {e}")
                self.records = []

    def _save_records(self):
        """Save records to JSON file."""
        try:
            with open(self.records_file, 'w') as f:
                json.dump(self.records, f, indent=2)
        except Exception as e:
            logger.error(f"Error saving records: {e}")

    def add_detection(self, timestamp, vehicle_img, plate_img=None,
                      plate_number=None, confidence=0.0, frame_number=None):
        """
        Add a new vehicle detection, grouping with existing detections if within time window.

        Args:
            timestamp: Detection time
            vehicle_img: Image of the detected vehicle
            plate_img: Image of the license plate (or None if not detected)
            plate_number: Recognized license plate text (or None if not recognized)
            confidence: Confidence score for the plate recognition
            frame_number: Optional frame number (for video file processing)
        """
        # Generate unique filename for the vehicle image
        img_id = str(uuid.uuid4())
        vehicle_filename = f"{img_id}_vehicle.jpg"
        plate_filename = f"{img_id}_plate.jpg" if plate_img is not None else None

        # Save vehicle image
        vehicle_path = os.path.join(self.storage_path, "images", vehicle_filename)
        cv2.imwrite(vehicle_path, vehicle_img)

        # Save plate image if available
        plate_path = None
        if plate_img is not None:
            plate_path = os.path.join(self.storage_path, "images", plate_filename)
            cv2.imwrite(plate_path, plate_img)

        # Create detection record
        detection = {
            "timestamp": timestamp,
            "datetime": datetime.fromtimestamp(timestamp).strftime("%Y-%m-%d %H:%M:%S"),
            "vehicle_img": vehicle_filename,
            "plate_img": plate_filename,
            "plate_number": plate_number,
            "confidence": confidence
        }

        # Add frame number if provided (for video file processing)
        if frame_number is not None:
            detection["frame_number"] = frame_number

        # Try to group with existing active groups
        grouped = self._try_group_detection(detection)

        if not grouped:
            # Create a new group
            group_id = str(uuid.uuid4())

            new_group = {
                "id": group_id,
                "first_seen": timestamp,
                "last_seen": timestamp,
                "detections": [detection],
                "best_plate": plate_number,
                "best_confidence": confidence if plate_number else 0.0
            }

            # Add to active groups
            self.active_groups[group_id] = new_group

        # Clean up expired groups
        self._cleanup_expired_groups(timestamp)

        # Ensure we don't exceed max images
        self._enforce_storage_limit()

    def _try_group_detection(self, detection):
        """
        Try to group detection with existing active groups.

        Args:
            detection: Detection record

        Returns:
            Boolean indicating if grouping was successful
        """
        current_time = detection["timestamp"]
        plate_number = detection["plate_number"]
        confidence = detection["confidence"]

        # First, try to match by plate number if available
        if plate_number:
            for group_id, group in self.active_groups.items():
                if group["best_plate"] == plate_number:
                    # Match found, add to group
                    group["detections"].append(detection)
                    group["last_seen"] = current_time

                    # Update best confidence if this one is better
                    if confidence > group["best_confidence"]:
                        group["best_confidence"] = confidence

                    return True

        # If no match by plate or no plate available, match by time proximity
        # This is a simplified approach - in a real implementation,
        # you might want to use visual similarity or location

        for group_id, group in self.active_groups.items():
            # If within time window, consider it the same vehicle
            if current_time - group["last_seen"] < self.grouping_window:
                group["detections"].append(detection)
                group["last_seen"] = current_time

                # Update best plate if this detection has one and it's better
                if plate_number and (not group["best_plate"] or confidence > group["best_confidence"]):
                    group["best_plate"] = plate_number
                    group["best_confidence"] = confidence

                return True

        return False

    def _cleanup_expired_groups(self, current_time):
        """
        Move expired groups from active tracking to permanent records.

        Args:
            current_time: Current timestamp
        """
        expired_groups = []

        for group_id, group in self.active_groups.items():
            # If no activity for more than grouping window, consider it complete
            if current_time - group["last_seen"] > self.grouping_window:
                expired_groups.append(group_id)

                # Create a record from this group
                record = {
                    "id": group["id"],
                    "first_seen": group["first_seen"],
                    "last_seen": group["last_seen"],
                    "datetime": datetime.fromtimestamp(group["first_seen"]).strftime("%Y-%m-%d %H:%M:%S"),
                    "plate_number": group["best_plate"],
                    "confidence": group["best_confidence"],
                    "num_detections": len(group["detections"]),
                    "images": [d["vehicle_img"] for d in group["detections"]],
                    "plate_images": [d["plate_img"] for d in group["detections"] if d["plate_img"]]
                }

                # Add to permanent records
                self.records.append(record)

                # Log the new record
                if record["plate_number"]:
                    plate_info = f"with plate {record['plate_number']}"
                else:
                    plate_info = "without plate recognition"

                logger.info(f"New vehicle record created {plate_info} with {record['num_detections']} images")

        # Remove expired groups
        for group_id in expired_groups:
            del self.active_groups[group_id]

        # If any groups were expired, save records
        if expired_groups:
            self._save_records()

    def _enforce_storage_limit(self):
        """Ensure storage doesn't exceed maximum limit."""
        # Count image files
        image_dir = os.path.join(self.storage_path, "images")
        image_files = os.listdir(image_dir)

        if len(image_files) <= self.max_stored_images:
            return

        # Need to remove some old images
        excess = len(image_files) - self.max_stored_images

        # Sort records by timestamp (oldest first)
        self.records.sort(key=lambda x: x["first_seen"])

        removed_records = []
        removed_images = 0

        # Remove oldest records until we've freed up enough space
        for record in self.records:
            # Count images in this record
            record_images = record["images"] + record["plate_images"]

            # Remove images
            for img_file in record_images:
                if img_file:
                    try:
                        os.remove(os.path.join(image_dir, img_file))
                        removed_images += 1
                    except Exception as e:
                        logger.warning(f"Failed to remove image {img_file}: {e}")

            removed_records.append(record)

            if removed_images >= excess:
                break

        # Update records list
        for record in removed_records:
            self.records.remove(record)

        # Save updated records
        self._save_records()

        logger.info(f"Storage limit enforced: removed {len(removed_records)} old records and {removed_images} images")

    def get_latest_records(self, count=10):
        """
        Get the most recent vehicle records.

        Args:
            count: Number of records to return

        Returns:
            List of the most recent vehicle records
        """
        # Sort by last_seen timestamp (newest first)
        sorted_records = sorted(self.records, key=lambda x: x["last_seen"], reverse=True)
        return sorted_records[:count]
