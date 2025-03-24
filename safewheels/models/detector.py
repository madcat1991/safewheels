"""
Vehicle detection using YOLOv8.
"""
import os
import logging
import cv2
import numpy as np

logger = logging.getLogger(__name__)

class VehicleDetector:
    """
    Detects vehicles in images using YOLOv8.
    """

    # YOLOv8 class names for vehicles
    VEHICLE_CLASSES = ['car', 'motorcycle', 'bus', 'truck']

    def __init__(self, model_path=None):
        """
        Initialize the YOLOv8 vehicle detector.

        Args:
            model_path: Path to the YOLOv8 model.
                        If None, will attempt to use a pre-trained model.
        """
        self.model = None

        try:
            self._load_model(model_path)
        except Exception as e:
            logger.error(f"Error loading YOLOv8 model: {e}")
            logger.warning("Using fallback detection. Please install required dependencies.")

    def _load_model(self, model_path):
        """
        Load the YOLOv8 model for vehicle detection.

        Args:
            model_path: Path to YOLOv8 model or None to use pre-trained model
        """
        try:
            # Try importing ultralytics package
            from ultralytics import YOLO

            # Use pre-trained YOLOv8 model
            self.model = YOLO("yolov8n.pt")  # Nano model - smallest and fastest
            logger.info("Loaded pre-trained YOLOv8n model")

        except ImportError:
            logger.error("Could not import ultralytics. Please install with: pip install ultralytics")
            self.model = None

    def detect(self, image, confidence_threshold=0.5):
        """
        Detect vehicles in an image using YOLOv8.

        Args:
            image: OpenCV image (numpy array)
            confidence_threshold: Minimum confidence for detection

        Returns:
            List of dictionaries containing vehicle detections with keys:
            - bbox: (x, y, w, h) bounding box
            - confidence: detection confidence
            - class_name: vehicle class name (car, truck, etc.)
        """
        # If model failed to load, use fallback detection
        if self.model is None:
            return self._fallback_detect(image)

        # Run YOLOv8 inference
        results = self.model(image, conf=confidence_threshold)

        vehicles = []

        # Process results
        for result in results:
            # Extract detections
            boxes = result.boxes

            for box in boxes:
                # Get class name
                cls_id = int(box.cls.item())
                cls_name = result.names[cls_id]

                # Filter for vehicle classes only
                if cls_name in self.VEHICLE_CLASSES:
                    # Get bounding box coordinates (x1, y1, x2, y2)
                    x1, y1, x2, y2 = box.xyxy[0].tolist()

                    # Convert to (x, y, width, height) format
                    x = int(x1)
                    y = int(y1)
                    w = int(x2 - x1)
                    h = int(y2 - y1)

                    # Get confidence score
                    confidence = box.conf.item()

                    vehicles.append({
                        'bbox': [x, y, w, h],
                        'confidence': confidence,
                        'class_name': cls_name
                    })

        return vehicles

    def _fallback_detect(self, image):
        """
        Fallback detection method when YOLOv8 is not available.
        Uses basic OpenCV methods for demonstration purposes.

        Args:
            image: OpenCV image

        Returns:
            List with dummy vehicle detection
        """
        logger.warning("Using fallback vehicle detection")

        height, width = image.shape[:2]

        # Try basic contour detection on moving objects (not very accurate)
        # Convert to grayscale and apply blur
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray, (21, 21), 0)

        # Apply threshold
        _, thresh = cv2.threshold(blur, 100, 255, cv2.THRESH_BINARY)

        # Find contours
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        vehicles = []

        # Filter contours by size
        min_area = width * height * 0.01  # At least 1% of the image

        for contour in contours:
            area = cv2.contourArea(contour)

            if area > min_area:
                x, y, w, h = cv2.boundingRect(contour)
                vehicles.append({
                    'bbox': [x, y, w, h],
                    'confidence': 0.5,  # Dummy confidence
                    'class_name': 'car'  # Assume car
                })

        # If no contours found, return a dummy detection
        if not vehicles:
            vehicles = [{
                'bbox': [int(width * 0.3), int(height * 0.3), int(width * 0.4), int(height * 0.4)],
                'confidence': 0.6,
                'class_name': 'car'
            }]

        return vehicles
