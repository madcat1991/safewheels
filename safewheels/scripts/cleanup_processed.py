#!/usr/bin/env python
"""
Cleanup script that deletes unused vehicle detection images and their database records.

This script helps manage disk space by removing detections that don't have
corresponding recognition records, along with their associated images,
while preserving all detections referenced by recognitions.
"""
import os
import sqlite3
import argparse
import logging
import time

from safewheels.utils.config import load_config

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("cleanup_processed")


class CleanupProcessor:
    """
    Finds and removes images and database records for detections
    that don't have corresponding recognition records.
    """

    def __init__(self, config_path, min_age_hours=24, dry_run=False):
        """
        Initialize the cleanup processor.

        Args:
            config_path: Path to the configuration file
            min_age_hours: Minimum age of records to consider for cleanup (in hours)
            dry_run: If True, only log actions without actually deleting files or records
        """
        # Load configuration
        self.config = load_config(config_path)
        self.storage_path = self.config.get("storage_path")
        self.min_age_hours = min_age_hours
        self.dry_run = dry_run

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

        # Check if images directory exists
        if not os.path.exists(self.images_dir):
            logger.error(f"Images directory {self.images_dir} doesn't exist")
            raise FileNotFoundError(f"Images directory {self.images_dir} not found")

        logger.info(f"Cleanup processor initialized with database: {self.db_path}")
        logger.info(f"Images directory: {self.images_dir}")
        logger.info(f"Minimum age for cleanup: {min_age_hours} hours")
        logger.info(f"Dry run mode: {dry_run}")

    def get_unused_detections(self):
        """
        Get all detections that:
        1. Don't have a corresponding recognition record
        2. Belong to a vehicle that has at least one successful recognition
        3. Are older than min_age_hours

        Returns:
            List of detection records (id, image) that can be cleaned up
        """
        try:
            conn = sqlite3.connect(self.db_path)
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()

            # Calculate the timestamp cutoff
            current_time = time.time()
            cutoff_time = current_time - (self.min_age_hours * 3600)  # Convert hours to seconds

            # Query to find unused detections meeting criteria
            query = """
                SELECT d.id, d.image
                FROM detections d
                    LEFT JOIN recognitions r ON d.id = r.detection_id
                WHERE d.timestamp < ? AND r.id IS NULL
            """

            cursor.execute(query, (cutoff_time,))
            unused_detections = [dict(row) for row in cursor.fetchall()]

            conn.close()

            logger.info(f"Found {len(unused_detections)} unused detections older than {self.min_age_hours} hours")
            return unused_detections

        except sqlite3.Error as e:
            logger.error(f"Error querying database: {e}")
            return []

    def run_cleanup(self):
        """
        Run the cleanup process for all eligible unused detections in batch mode.

        Returns:
            Tuple of (detections_processed, images_deleted, records_deleted)
        """
        unused_detections = self.get_unused_detections()

        if not unused_detections:
            logger.info("No unused detections found to clean up")
            return 0, 0, 0

        # Extract detection IDs and image filenames
        detection_ids, image_filenames = [], []
        for detection in unused_detections:
            detection_ids.append((detection["id"],))  # SQLite expects a tuple
            image_filenames.append(detection["image"])

        images_deleted = 0
        records_deleted = 0

        # Batch delete detection records
        if not self.dry_run and detection_ids:
            try:
                conn = sqlite3.connect(self.db_path)
                cursor = conn.cursor()

                cursor.executemany("DELETE FROM detections WHERE id = ?", detection_ids)
                records_deleted = cursor.rowcount

                conn.commit()
                conn.close()

                logger.info(f"Deleted {records_deleted} detection records in batch")
            except sqlite3.Error as e:
                logger.error(f"Error batch deleting detection records: {e}")
        elif self.dry_run:
            # In dry run mode, just count what would be deleted
            records_deleted = len(detection_ids)
            logger.info(f"Would delete {records_deleted} detection records in batch")

        # Delete images only if detection records were deleted
        if records_deleted > 0:
            # Batch delete images
            for image_filename in image_filenames:
                image_path = os.path.join(self.images_dir, image_filename)
                if os.path.exists(image_path):
                    if not self.dry_run:
                        try:
                            os.remove(image_path)
                            images_deleted += 1
                        except OSError as e:
                            logger.error(f"Error deleting image {image_path}: {e}")
                    else:
                        logger.debug(f"Would delete image: {image_path}")
                        images_deleted += 1
                else:
                    logger.warning(f"Image file not found: {image_path}")

        action_word = "Would delete" if self.dry_run else "Deleted"
        logger.info(f"Cleanup completed: {action_word} {records_deleted} detection records and {images_deleted} images")

        return len(unused_detections), images_deleted, records_deleted


def main():
    # Parse command-line arguments
    parser = argparse.ArgumentParser(
        description="Clean up unused vehicle detection records and their images"
    )
    parser.add_argument("-c", "--config", required=True, help="Path to the configuration file")
    parser.add_argument(
        "--min-age",
        type=float,
        default=24.0,
        help="Minimum age in hours for records to be eligible for cleanup (default: 24)"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Run in dry-run mode (don't actually delete files)"
    )
    args = parser.parse_args()

    # Initialize and run the cleanup processor
    try:
        processor = CleanupProcessor(
            config_path=args.config,
            min_age_hours=args.min_age,
            dry_run=args.dry_run
        )

        processor.run_cleanup()

    except Exception as e:
        logger.error(f"Error during cleanup: {e}")
        return 1

    return 0


if __name__ == "__main__":
    exit(main())
