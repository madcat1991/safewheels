#!/usr/bin/env python
"""
RTSP stream processing for vehicle and license plate detection.
"""
import time
import logging
import argparse

from safewheels.functions import get_compute_params
from safewheels.stream_processor import StreamProcessor
from safewheels.models.vehicle_detector import VehicleDetector
from safewheels.models.plate_detector import PlateDetector
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
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Vehicle and license plate detection script')
    parser.add_argument('--config', type=str, help='Path to configuration file')
    args = parser.parse_args()

    # Load configuration
    config = load_config(args.config)

    # Get RTSP URL from configuration
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

    input_source = rtsp_url
    logger.info(f"Using RTSP stream: {input_source}")

    use_gpu, device, model_precision = get_compute_params(config)

    vehicle_detector = VehicleDetector(
        use_gpu=use_gpu,
        device=device,
        model_precision=model_precision
    )

    plate_detector = PlateDetector(
        gpu=use_gpu,
        device=device,
        model_precision=model_precision
    )

    record_manager = RecordManager(
        storage_path=config.get('storage_path', 'data/vehicles'),
        vehicle_id_threshold_sec=config.get('vehicle_id_threshold_sec', 5),
        db_filename=config.get('db_filename', 'detections.db'),
        images_dirname=config.get('images_dirname', 'images')
    )

    # Create stream processor
    processor = StreamProcessor(
        input_source=input_source,
        vehicle_detector=vehicle_detector,
        plate_detector=plate_detector,
        record_manager=record_manager,
        config=config
    )

    try:
        logger.info("Starting SafeWheels monitoring")
        processor.start()

        # Keep running indefinitely for the stream
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
