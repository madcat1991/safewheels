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
import av

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
        self.confidence_threshold = config.get('confidence_threshold', 0.5)
        # Lowered threshold for improved recall
        self.plate_confidence_threshold = config.get('plate_confidence_threshold', 0.3)
        # Process every nth frame (default: 10)
        self.process_every_n_frames = config.get('process_every_n_frames', 10)

        # Thread control
        self._running = False
        self.thread = None

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

    def _process_video(self):
        """Main processing loop for the video source (stream or file) using PyAV."""
        source_type = "video file" if self.is_file else "RTSP stream"
        logger.info(f"Opening {source_type}: {self.input_source}")

        while self._running:
            try:
                # For RTSP streams, set options before opening
                options = {}
                if not self.is_file:
                    options = {'rtsp_transport': 'udp', 'timeout': '5000000'}  # 5 seconds in microseconds

                # Use PyAV to process the video
                with av.open(self.input_source, options=options) as container:
                    video_stream = container.streams.video[0]
                    video_stream.thread_type = 'AUTO'  # Enable multithreading

                    # Get video properties
                    fps = getattr(video_stream, 'average_rate', None) or video_stream.framerate
                    total_frames = video_stream.frames if video_stream.frames > 0 else None

                    if self.is_file and total_frames:
                        logger.info(
                            f"Video has {total_frames} frames at {fps} FPS. "
                            f"Processing every {self.process_every_n_frames}th frame."
                        )
                    else:
                        logger.info(f"Stream FPS: {fps}. Processing every {self.process_every_n_frames}th frame.")

                    frame_count = 0
                    frames_processed = 0

                    for frame in container.decode(video=0):
                        if not self._running:
                            break

                        frame_count += 1

                        # Process every nth frame (prioritizing I-frames when possible)
                        if frame.key_frame or frame_count % self.process_every_n_frames == 0:
                            # Convert PyAV frame to OpenCV format
                            img = frame.to_ndarray(format='bgr24')

                            # Process the frame
                            try:
                                timestamp = time.time()
                                self._process_frame(img, timestamp, frame_count)
                                frames_processed += 1

                                frame_type = "I-frame" if frame.key_frame else "frame"
                                if self.is_file and total_frames:
                                    logger.info(
                                        f"Processed {frame_type} {frame_count}/{total_frames} (#{frames_processed})"
                                    )
                                else:
                                    logger.info(f"Processed {frame_type} (#{frames_processed})")
                            except Exception as e:
                                logger.error(f"Error processing frame {frame_count}: {e}")

                    # If we're processing a file and reached the end
                    if self.is_file:
                        logger.info(f"Finished processing file. Total frames processed: {frames_processed}")
                        self._running = False
                        break
                    else:
                        # For streams, if we get here, we need to reconnect
                        logger.warning("Stream ended. Attempting to reconnect...")
                        time.sleep(3)  # Wait before reconnecting

            except Exception as e:
                logger.error(f"Error in video processing: {e}")
                if self.is_file:
                    self._running = False
                    break
                else:
                    # For streams, try to reconnect
                    logger.warning(f"Connection error. Attempting to reconnect in 5 seconds...")
                    time.sleep(5)

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
            vehicle_img = self._sharpen_image(vehicle_img)

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

    def _sharpen_image(self, img):
        """
        Apply a simple unsharp masking to lightly sharpen the image.
        """
        gaussian = cv2.GaussianBlur(img, (9, 9), 10.0)
        return cv2.addWeighted(img, 1.5, gaussian, -0.5, 0)
