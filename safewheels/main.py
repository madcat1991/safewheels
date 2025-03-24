#!/usr/bin/env python
"""
Main module for SafeWheels - RTSP stream monitoring for vehicle detection and license plate recognition.
"""
import time
import logging

from safewheels.stream_processor import StreamProcessor
from safewheels.models.detector import VehicleDetector
from safewheels.models.plate_recognizer import PlateRecognizer
from safewheels.storage.record_manager import RecordManager
from safewheels.utils.config import load_config

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger(__name__)


def main():
    """Main execution function for SafeWheels."""
    # Load configuration
    config = load_config()
    rtsp_url = config.get('rtsp_url')
    rtsp_username = config.get('rtsp_username')
    rtsp_password = config.get('rtsp_password')

    if not rtsp_url:
        logger.error("No RTSP URL provided in configuration.")
        return

    # Build authenticated URL if credentials are provided
    if rtsp_username and rtsp_password:
        # Format: rtsp://username:password@ip:port/path
        parsed_url = rtsp_url.split('://')
        if len(parsed_url) == 2:
            protocol, address = parsed_url
            rtsp_url = f"{protocol}://{rtsp_username}:{rtsp_password}@{address}"
            logger.info("Using authenticated RTSP stream")

    # Initialize components
    vehicle_detector = VehicleDetector()
    plate_recognizer = PlateRecognizer()
    record_manager = RecordManager()

    # Create stream processor
    processor = StreamProcessor(
        rtsp_url=rtsp_url,
        vehicle_detector=vehicle_detector,
        plate_recognizer=plate_recognizer,
        record_manager=record_manager,
        config=config
    )

    try:
        logger.info("Starting SafeWheels monitoring")
        processor.start()

        # Keep the process running
        while True:
            time.sleep(1)

    except KeyboardInterrupt:
        logger.info("Shutting down SafeWheels")
        processor.stop()
    except Exception as e:
        logger.error(f"Error in main process: {e}")
        processor.stop()


if __name__ == "__main__":
    main()
