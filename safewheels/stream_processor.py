"""
RTSP stream processor for vehicle and license plate detection.
"""
import cv2
import os
import time
import threading
import logging
from datetime import datetime

logger = logging.getLogger(__name__)

class StreamProcessor:
    """
    Processes RTSP video stream to detect vehicles and recognize license plates.
    """
    def __init__(self, rtsp_url, vehicle_detector, plate_recognizer, record_manager, config):
        """
        Initialize StreamProcessor.
        
        Args:
            rtsp_url: RTSP stream URL with authentication if needed
            vehicle_detector: VehicleDetector instance
            plate_recognizer: PlateRecognizer instance
            record_manager: RecordManager for storing detection results
            config: Application configuration dictionary
        """
        self.rtsp_url = rtsp_url
        self.vehicle_detector = vehicle_detector
        self.plate_recognizer = plate_recognizer
        self.record_manager = record_manager
        self.config = config
        
        # Processing parameters
        self.detection_interval = config.get('detection_interval', 1.0)
        self.confidence_threshold = config.get('confidence_threshold', 0.5)
        self.plate_confidence_threshold = config.get('plate_confidence_threshold', 0.7)
        
        # Thread control
        self.running = False
        self.thread = None
        self.capture = None
    
    def start(self):
        """Start processing the RTSP stream in a separate thread."""
        if self.running:
            return
            
        self.running = True
        self.thread = threading.Thread(target=self._process_stream)
        self.thread.daemon = True
        self.thread.start()
    
    def stop(self):
        """Stop processing the RTSP stream."""
        self.running = False
        if self.thread:
            self.thread.join(timeout=3.0)
        if self.capture:
            self.capture.release()
            
    def _process_stream(self):
        """Main processing loop for the RTSP stream."""
        logger.info(f"Connecting to RTSP stream: {self.rtsp_url}")
        
        # OpenCV RTSP connection settings
        os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = "rtsp_transport;udp"
        
        self.capture = cv2.VideoCapture(self.rtsp_url, cv2.CAP_FFMPEG)
        
        if not self.capture.isOpened():
            logger.error("Failed to open RTSP stream")
            self.running = False
            return
        
        last_process_time = time.time()
        
        while self.running:
            # Throttle processing based on configured interval
            current_time = time.time()
            if current_time - last_process_time < self.detection_interval:
                time.sleep(0.1)  # Small sleep to reduce CPU usage
                continue
                
            last_process_time = current_time
            
            # Read frame
            ret, frame = self.capture.read()
            if not ret:
                logger.warning("Failed to read frame from stream")
                # Attempt to reconnect if connection was lost
                self.capture.release()
                time.sleep(2)  # Wait before reconnecting
                self.capture = cv2.VideoCapture(self.rtsp_url, cv2.CAP_FFMPEG)
                continue
            
            # Process current frame
            try:
                self._process_frame(frame, current_time)
            except Exception as e:
                logger.error(f"Error processing frame: {e}")
                
        # Clean up
        if self.capture:
            self.capture.release()
            
    def _process_frame(self, frame, timestamp):
        """
        Process a single video frame.
        
        Args:
            frame: OpenCV image frame
            timestamp: Current timestamp
        """
        # Detect vehicles
        vehicles = self.vehicle_detector.detect(frame, self.confidence_threshold)
        
        if not vehicles:
            return
            
        timestamp_str = datetime.fromtimestamp(timestamp).strftime('%Y-%m-%d %H:%M:%S')
        logger.info(f"Detected {len(vehicles)} vehicles at {timestamp_str}")
        
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
                confidence=confidence
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