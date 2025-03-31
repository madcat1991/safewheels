"""
Record manager for storing vehicle and license plate detections.
"""
import os
import logging
import cv2
import sqlite3
from datetime import datetime
import uuid

logger = logging.getLogger(__name__)


class RecordManager:
    """
    Storage manager for vehicle detections and license plate information.
    Saves detection data in SQLite database and manages image storage.
    """

    def __init__(self, storage_path, vehicle_id_threshold_sec, db_filename, images_dirname):
        """
        Initialize the record manager.

        Args:
            storage_path: Path to store detection images and metadata
            vehicle_id_threshold_sec: Time threshold in seconds to consider as the same vehicle
            db_filename: Name of the SQLite database file
            images_dirname: Name of the directory to store images
        """
        self.storage_path = storage_path
        self.vehicle_id_threshold_sec = vehicle_id_threshold_sec
        self.db_filename = db_filename
        self.images_dirname = images_dirname

        # Vehicle tracking variables
        self.last_detection_time = 0
        self.current_vehicle_id = self._generate_vehicle_id()

        # Ensure storage directory exists
        os.makedirs(storage_path, exist_ok=True)
        os.makedirs(os.path.join(storage_path, images_dirname), exist_ok=True)

        # Setup SQLite database
        self.db_path = os.path.join(storage_path, db_filename)

        # Check if database exists - if not, initialize it
        if not os.path.exists(self.db_path):
            self._init_database()
        else:
            # Just connect to verify the database is valid
            try:
                conn = sqlite3.connect(self.db_path)
                conn.close()
            except sqlite3.Error as e:
                logger.error(f"Error connecting to database: {e}")
                raise ValueError(f"Could not connect to existing database: {e}")

        logger.info(
            f"Record manager initialized with storage at {storage_path}, "
            f"database at {self.db_path}, "
            f"vehicle ID threshold: {vehicle_id_threshold_sec}s"
        )

    def _init_database(self):
        """Initialize the SQLite database with the necessary tables."""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()

            # Create detections table
            cursor.execute('''
            CREATE TABLE detections (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                vehicle_id TEXT NOT NULL,
                timestamp REAL NOT NULL,
                datetime_ms TEXT NOT NULL,
                image TEXT NOT NULL,
                vehicle_confidence REAL NOT NULL,
                plate_detection_confidence REAL NOT NULL,
                plate_number TEXT,
                ocr_confidence REAL NOT NULL,
                frame_number INTEGER
            )
            ''')

            # Create indexes for faster queries
            cursor.execute('CREATE INDEX idx_vehicle_id ON detections(vehicle_id)')
            cursor.execute('CREATE INDEX idx_timestamp ON detections(timestamp)')

            conn.commit()
            conn.close()
            logger.info(f"Created new SQLite database at {self.db_path}")
        except sqlite3.Error as e:
            logger.error(f"Error creating database: {e}")
            raise ValueError(f"Failed to initialize database: {e}")

    def _generate_vehicle_id(self):
        """Generate a unique short hash ID for a vehicle."""
        return uuid.uuid4().hex[:6]  # 6 character hex ID

    def add_detection(self, timestamp, vehicle_img, vehicle_confidence=0.0, plate_bbox=None,
                      plate_detection_confidence=0.0, plate_number=None, ocr_confidence=0.0, frame_number=None):
        """
        Add a new vehicle detection record and save the image with a bounding box for the license plate.

        Args:
            timestamp: Detection time
            vehicle_img: Image of the detected vehicle
            vehicle_confidence: Confidence score for the vehicle detection
            plate_bbox: Bounding box of the detected license plate (x, y, w, h) or None
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

        # Create a copy of the vehicle image to draw on
        annotated_img = vehicle_img.copy()

        # Draw bounding box for license plate if available
        if plate_bbox is not None:
            x, y, w, h = plate_bbox

            # Draw green rectangle around the plate
            cv2.rectangle(
                annotated_img,
                (x, y),
                (x + w, y + h),
                (0, 255, 0),  # Green color
                2  # Line thickness
            )

            # Add the detected plate number as text if available
            if plate_number:
                # Position the text above the bounding box if there's space
                text_y = max(y - 10, 10)
                cv2.putText(
                    annotated_img,
                    plate_number,
                    (x, text_y),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,  # Font scale
                    (0, 255, 0),  # Green color
                    2  # Line thickness
                )

        # Generate filename with timestamp (no "vehicle" or "car" in the name)
        current_time = datetime.now()
        timestamp_ms = current_time.strftime("%Y%m%d_%H%M%S_%f")  # Full timestamp with microseconds
        image_filename = f"{timestamp_ms}.jpg"

        # Save the annotated image
        image_path = os.path.join(self.storage_path, self.images_dirname, image_filename)
        cv2.imwrite(image_path, annotated_img)

        # Insert detection record into database
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()

            cursor.execute('''
            INSERT INTO detections (
                vehicle_id, timestamp, datetime_ms, image,
                vehicle_confidence, plate_detection_confidence,
                plate_number, ocr_confidence, frame_number
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                self.current_vehicle_id,
                timestamp,
                datetime.fromtimestamp(timestamp).strftime("%Y-%m-%d %H:%M:%S.%f"),
                image_filename,
                vehicle_confidence,
                plate_detection_confidence,
                plate_number if plate_number else "",
                ocr_confidence,
                frame_number
            ))

            conn.commit()
            conn.close()
        except sqlite3.Error as e:
            logger.error(f"Error saving to database: {e}")
            return

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
