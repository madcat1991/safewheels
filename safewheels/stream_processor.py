"""
Stream and video file processor for vehicle and license plate detection.
"""
import cv2
import os
import time
import threading
import logging
from datetime import datetime
import os.path

logger = logging.getLogger(__name__)


class StreamProcessor:
    """
    Processes video streams and files to detect vehicles and recognize license plates.
    """
    def __init__(self, input_source, vehicle_detector, plate_recognizer, record_manager, config):
        """
        Initialize StreamProcessor.

        Args:
            input_source: RTSP stream URL or video file path
            vehicle_detector: VehicleDetector instance
            plate_recognizer: PlateRecognizer instance
            record_manager: RecordManager for storing detection results
            config: Application configuration dictionary
        """
        self.input_source = input_source
        self.vehicle_detector = vehicle_detector
        self.plate_recognizer = plate_recognizer
        self.record_manager = record_manager
        self.config = config

        # Determine if source is a video file or a stream
        self.is_file = os.path.isfile(input_source) if isinstance(input_source, str) else False

        # Processing parameters
        self.detection_interval = config.get('detection_interval', 1.0)  # 1 frame per second
        self.confidence_threshold = config.get('confidence_threshold', 0.5)
        self.plate_confidence_threshold = config.get('plate_confidence_threshold', 0.3)  # Lowered threshold for improved recall

        # Thread control
        self._running = False
        self.thread = None
        self.capture = None
        
    @property
    def running(self):
        """Check if the processor is currently running."""
        return self._running

    def start(self):
        """Start processing the video source in a separate thread."""
        if self._running:
            return

        self._running = True
        self.thread = threading.Thread(target=self._process_video)
        self.thread.daemon = True
        self.thread.start()

    def stop(self):
        """Stop processing the video source."""
        self._running = False
        if self.thread:
            self.thread.join(timeout=3.0)
        if self.capture:
            self.capture.release()

    def _process_video(self):
        """Main processing loop for the video source (stream or file)."""
        source_type = "video file" if self.is_file else "RTSP stream"
        logger.info(f"Opening {source_type}: {self.input_source}")

        # Set up video capture
        if self.is_file:
            self.capture = cv2.VideoCapture(self.input_source)
            # Get video properties for file mode
            fps = self.capture.get(cv2.CAP_PROP_FPS)
            total_frames = int(self.capture.get(cv2.CAP_PROP_FRAME_COUNT))
            
            # Calculate frame skip based on desired frame rate
            # We want 1 frame per self.detection_interval seconds
            # So we need to skip (fps * self.detection_interval - 1) frames between each processed frame
            frames_to_skip = int(fps * self.detection_interval) - 1
            if frames_to_skip < 0:
                frames_to_skip = 0
                
            logger.info(f"Video has {total_frames} frames at {fps} FPS. Processing 1 frame every {self.detection_interval} seconds (skipping {frames_to_skip} frames between each).")
        else:
            # OpenCV RTSP connection settings
            os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = "rtsp_transport;udp"
            self.capture = cv2.VideoCapture(self.input_source, cv2.CAP_FFMPEG)

        if not self.capture.isOpened():
            logger.error(f"Failed to open {source_type}")
            self._running = False
            return

        last_process_time = time.time()
        frame_count = 0
        frames_since_last_process = 0

        while self._running:
            # For streaming mode, throttle by time
            if not self.is_file:
                current_time = time.time()
                if current_time - last_process_time < self.detection_interval:
                    time.sleep(0.1)  # Small sleep to reduce CPU usage
                    continue
                last_process_time = current_time

            # Read frame
            ret, frame = self.capture.read()
            if not ret:
                if self.is_file:
                    logger.info("Reached end of video file")
                    self._running = False
                    break
                else:
                    logger.warning("Failed to read frame from stream")
                    # Attempt to reconnect if connection was lost
                    self.capture.release()
                    time.sleep(2)  # Wait before reconnecting
                    self.capture = cv2.VideoCapture(self.input_source, cv2.CAP_FFMPEG)
                    continue

            frame_count += 1
            
            # For file mode, throttle by skipping frames
            if self.is_file:
                frames_since_last_process += 1
                # Only process every N frames
                if frames_to_skip > 0 and frames_since_last_process <= frames_to_skip:
                    continue
                frames_since_last_process = 0

            # Process current frame
            try:
                self._process_frame(frame, time.time(), frame_count)
                if self.is_file:
                    logger.info(f"Processed frame {frame_count}/{total_frames}")
            except Exception as e:
                logger.error(f"Error processing frame {frame_count}: {e}")

        # Clean up
        if self.capture:
            self.capture.release()

    def _process_frame(self, frame, timestamp, frame_count=None):
        """
        Process a single video frame.

        Args:
            frame: OpenCV image frame
            timestamp: Current timestamp
            frame_count: Frame number (for video files)
        """
        # Detect vehicles
        vehicles = self.vehicle_detector.detect(frame, self.confidence_threshold)

        if not vehicles:
            return

        timestamp_str = datetime.fromtimestamp(timestamp).strftime('%Y-%m-%d %H:%M:%S')
        frame_info = f" (frame {frame_count})" if frame_count is not None else ""
        logger.info(f"Detected {len(vehicles)} vehicles at {timestamp_str}{frame_info}")

        # Process each detected vehicle
        for i, vehicle in enumerate(vehicles):
            # Extract the vehicle portion of the image
            vehicle_img = self._crop_vehicle(frame, vehicle)

            # Attempt license plate recognition
            plate_img, plate_number, confidence = self.plate_recognizer.recognize(
                vehicle_img,
                self.plate_confidence_threshold
            )

            # Store the detection
            self.record_manager.add_detection(
                timestamp=timestamp,
                vehicle_img=vehicle_img,
                plate_img=plate_img if plate_img is not None else None,
                plate_number=plate_number,
                confidence=confidence,
                frame_number=frame_count
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
