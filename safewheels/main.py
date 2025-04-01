#!/usr/bin/env python
"""
Main module for SafeWheels - video stream and file processing for vehicle detection and license plate recognition.
"""
import time
import logging
import argparse
import os.path

from safewheels.stream_processor import StreamProcessor
from safewheels.models.detector import VehicleDetector
from safewheels.models.plate_recognizer import DePlateRecognizer
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
    parser = argparse.ArgumentParser(description='SafeWheels - Vehicle and license plate detection')
    parser.add_argument('--video', type=str, help='Path to video file for processing')
    parser.add_argument('--config', type=str, help='Path to configuration file')
    args = parser.parse_args()

    # Load configuration
    config = load_config(args.config)

    # Determine input source (video file or RTSP stream)
    if args.video and os.path.isfile(args.video):
        input_source = args.video
        logger.info(f"Using video file: {input_source}")
    else:
        # Use RTSP stream from config if no video file provided
        rtsp_url = config.get('rtsp_url')
        rtsp_username = config.get('rtsp_username')
        rtsp_password = config.get('rtsp_password')

        if not rtsp_url:
            logger.error("No RTSP URL provided in configuration and no valid video file specified.")
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

    # Initialize components with GPU configuration
    use_gpu = config.get('use_gpu', True)
    device_type = config.get('device_type', None)  # Auto-detect if None
    model_precision = config.get('model_precision', 'fp16')
    cuda_device = config.get('cuda_device', 0)

    # If CUDA device is specified, update device name
    device = None
    if device_type == 'cuda' and cuda_device is not None:
        device = f'cuda:{cuda_device}'
    else:
        device = device_type

    logger.info(f"GPU configuration: use_gpu={use_gpu}, device={device}, precision={model_precision}")

    vehicle_detector = VehicleDetector(
        use_gpu=use_gpu,
        device=device,
        model_precision=model_precision
    )

    plate_recognizer = DePlateRecognizer(
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
        plate_recognizer=plate_recognizer,
        record_manager=record_manager,
        config=config
    )

    try:
        logger.info("Starting SafeWheels monitoring")
        processor.start()

        # For video files, wait until processing is done
        # For streams, keep the process running indefinitely
        if args.video and os.path.isfile(args.video):
            # Wait for video processing to finish by checking if processor is still running
            while processor.running:
                time.sleep(1)
            logger.info("Video processing completed")
        else:
            # For streams, keep running indefinitely
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
