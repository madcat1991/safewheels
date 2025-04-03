"""
Stream processor for vehicle and license plate detection.
"""
import time
import threading
import logging
from datetime import datetime
import av

from safewheels.models.vehicle_detector import VehicleDetector
from safewheels.models.plate_detector import PlateDetector
from safewheels.storage.record_manager import RecordManager

logger = logging.getLogger(__name__)


class StreamProcessor:
    """
    Processes RTSP streams to detect vehicles and their license plates.
    """
    def __init__(
            self,
            input_source,
            vehicle_detector: VehicleDetector,
            plate_detector: PlateDetector,
            record_manager: RecordManager,
            config
            ):
        """
        Initialize StreamProcessor.

        Args:
            input_source: RTSP stream URL
            vehicle_detector: VehicleDetector instance
            plate_recognizer: PlateDetector instance
            record_manager: RecordManager for storing detection results
            config: Application configuration dictionary
        """
        self.input_source = input_source
        self.vehicle_detector = vehicle_detector
        self.plate_detector = plate_detector
        self.record_manager = record_manager
        self.config = config

        # Processing parameters - thresholds for each stage of detection
        self.vehicle_confidence_threshold = config.get('vehicle_confidence_threshold', 0.4)
        self.plate_detection_threshold = config.get('plate_detection_threshold', 0.3)

        # Process every nth frame
        self.process_every_n_frames = config.get('process_every_n_frames', 5)

        logger.info(
            f"Detection thresholds: vehicle={self.vehicle_confidence_threshold}, "
            f"license_plate={self.plate_detection_threshold}"
        )

        # Thread control
        self._running = False
        self.thread = None

    @property
    def running(self):
        """Check if the processor is currently running."""
        return self._running

    def start(self):
        """Start processing the stream in a separate thread."""
        if self._running:
            return

        self._running = True
        self.thread = threading.Thread(target=self._process_stream)
        self.thread.daemon = True
        self.thread.start()

    def stop(self):
        """Stop processing the stream."""
        self._running = False
        if self.thread:
            self.thread.join(timeout=3.0)

    def _process_stream(self):
        """Main processing loop for the RTSP stream using PyAV."""
        logger.info(f"Opening RTSP stream: {self.input_source}")

        while self._running:
            try:
                # Options for RTSP streams:
                # - rtsp_transport: Use 'tcp' for more reliable streams over 'udp' if experiencing issues
                # - timeout: Timeout in microseconds
                # - max_delay: Maximum delay in microseconds before frames are dropped
                # - skip_frame: Skip frames in case of decoding errors
                options = {
                    # Use the transport protocol that works best (will auto-switch if errors persist)
                    'rtsp_transport': getattr(self, '_rtsp_transport', 'tcp'),
                    'stimeout': '5000000',  # Socket timeout in microseconds (5 seconds)
                    'max_delay': '500000',  # 500ms max delay
                    'skip_frame': '1',  # Skip non-reference frames on errors
                    # Don't buffer frames, generate timestamps if missing
                    'fflags': 'nobuffer+genpts+discardcorrupt',
                    'flags': 'low_delay',  # Low latency mode
                    'analyzeduration': '1000000'  # Analyze duration in microseconds (1 second)
                }

                # Use PyAV to process the video
                with av.open(self.input_source, options=options) as container:
                    video_stream = container.streams.video[0]
                    video_stream.thread_type = 'AUTO'  # Enable multithreading

                    # Log codec information for debugging
                    codec_name = video_stream.codec_context.name
                    logger.info(
                        f"Video codec: {codec_name}, "
                        f"width: {video_stream.width}, height: {video_stream.height}"
                    )

                    # Get video properties
                    fps = getattr(video_stream, 'average_rate', None) or video_stream.framerate
                    logger.info(f"Stream FPS: {fps}. Processing every {self.process_every_n_frames}th frame.")

                    # Counter for processing every nth frame
                    nth_frame_counter = 0

                    # Use non-strict decoding to handle problematic H.265 streams
                    for packet in container.demux(video=0):
                        if not self._running:
                            break

                        if packet.dts is None:
                            # Skip packets without timestamps
                            continue

                        try:
                            for frame in packet.decode():
                                if not self._running:
                                    break

                                # Use modulo counter to track every nth frame
                                nth_frame_counter = (nth_frame_counter + 1) % self.process_every_n_frames

                                # Process key frames and every nth frame
                                if frame.key_frame or nth_frame_counter == 0:
                                    try:
                                        # Convert PyAV frame to OpenCV format
                                        img = frame.to_ndarray(format='bgr24')

                                        # Process the frame
                                        timestamp = time.time()
                                        self._process_frame(img, timestamp)

                                        frame_type = "I-frame" if frame.key_frame else "frame"
                                        logger.debug(f"Processed {frame_type}")
                                    except Exception as e:
                                        logger.error(f"Error processing frame: {e}")
                        except av.AVError as e:
                            # Handle decoding errors for this packet but continue with next
                            logger.warning(f"Decoding error: {e} - skipping packet")

                    # If we get here, we need to reconnect
                    logger.warning("Stream ended. Attempting to reconnect...")
                    time.sleep(3)  # Wait before reconnecting

            except av.AVError as e:
                # For streams, try to reconnect with increasing backoff
                retry_delay = min(10, (self._retry_count if hasattr(self, '_retry_count') else 0) + 3)
                self._retry_count = retry_delay - 2  # Store for next time

                if 'End of file' in str(e):
                    logger.warning(f"Stream ended. Attempting to reconnect in {retry_delay} seconds...")
                else:
                    logger.warning(f"RTSP connection error: {e}. Reconnecting in {retry_delay} seconds...")

                # Try switching transport protocol on persistent errors
                # If we're using TCP, try UDP next time and vice versa
                if hasattr(self, '_rtsp_connection_errors'):
                    self._rtsp_connection_errors += 1
                    if self._rtsp_connection_errors > 3:
                        current_transport = options.get('rtsp_transport', 'tcp')
                        new_transport = 'udp' if current_transport == 'tcp' else 'tcp'
                        logger.info(f"Switching RTSP transport from {current_transport} to {new_transport}")
                        self._rtsp_transport = new_transport
                        self._rtsp_connection_errors = 0
                else:
                    self._rtsp_connection_errors = 1
                    self._rtsp_transport = options.get('rtsp_transport', 'tcp')

                time.sleep(retry_delay)
            except Exception as e:
                logger.error(f"Error in stream processing: {e}")
                # For streams, retry for general errors
                logger.warning(f"Unexpected error: {e}. Attempting to reconnect in 5 seconds...")
                time.sleep(5)

    def _is_good_aspect_ratio(self, vehicle_bbox, threshold=0.8):
        """
        Check if the vehicle image is suitable for processing based on its aspect ratio.

        Args:
            vehicle_img: Cropped vehicle image

        Returns:
            bool: True if the detection is valid, False otherwise
        """
        # Skip too vertical images (height significantly larger than width)
        x, y, w, h = vehicle_bbox
        aspect_ratio = w / h if h > 0 else 0
        return aspect_ratio >= threshold

    def _is_good_plate_position(self, vehicle_img, plate_bbox):
        """
        Check if the result of a detection is suitable for processing.

        Args:
            vehicle_img: Cropped vehicle image
            plate_bbox: License plate bounding box (x, y, w, h)

        Returns:
            bool: True if the detection is valid, False otherwise
        """
        h, w = vehicle_img.shape[:2]

        # Skip if plate bbox is too close to the vehicle image edge
        if plate_bbox:
            px, py, pw, ph = plate_bbox
            edge_margin = 0.05  # 5% margin from edges

            # Calculate distance from edges as percentage of vehicle image size
            left_edge_dist = px / w
            right_edge_dist = (w - (px + pw)) / w
            top_edge_dist = py / h
            bottom_edge_dist = (h - (py + ph)) / h

            # Skip if the plate is too close to any edge
            if (
                left_edge_dist < edge_margin or right_edge_dist < edge_margin or
                top_edge_dist < edge_margin or bottom_edge_dist < edge_margin
            ):
                logger.debug("Skipping vehicle with license plate too close to edge")
                return False

        return True

    def _process_frame(self, frame, timestamp):
        """
        Process a single video frame.

        Args:
            frame: OpenCV image frame
            timestamp: Current timestamp
        """
        # Detect vehicles
        vehicles = self.vehicle_detector.detect(frame, self.vehicle_confidence_threshold)

        if not vehicles:
            return

        timestamp_str = datetime.fromtimestamp(timestamp).strftime('%Y-%m-%d %H:%M:%S')
        logger.info(f"Detected {len(vehicles)} vehicles at {timestamp_str}")

        # Process each detected vehicle
        for i, vehicle in enumerate(vehicles):
            # skip vertical images
            if not self._is_good_aspect_ratio(vehicle['bbox']):
                continue

            # Extract the vehicle portion of the image
            vehicle_img = self._crop_vehicle(frame, vehicle)

            # Get vehicle detection confidence
            vehicle_confidence = vehicle['confidence']

            # Detect license plate using the detector part of the plate recognizer
            plate_bbox, plate_confidence = self.plate_detector.detect(
                vehicle_img,
                self.plate_detection_threshold
            )

            # skip images with bad plate positions
            if not self._is_good_plate_position(vehicle_img, plate_bbox):
                continue

            # Store the detection with detailed confidence values and bounding box
            self.record_manager.add_detection(
                timestamp=timestamp,
                vehicle_img=vehicle_img,
                vehicle_confidence=vehicle_confidence,
                plate_bbox=plate_bbox,
                plate_confidence=plate_confidence
            )

    def _crop_vehicle(self, frame, vehicle):
        """
        Crop the vehicle from the frame based on bounding box.

        Args:
            frame: Full image frame
            vehicle: Vehicle detection result with bounding box

        Returns:
            Cropped image of the vehicle
        """
        x, y, w, h = vehicle['bbox']

        # Ensure coordinates are within frame boundaries
        height, width = frame.shape[:2]
        x = max(0, int(x))
        y = max(0, int(y))
        w = min(width - x, int(w))
        h = min(height - y, int(h))

        return frame[y:y+h, x:x+w]
