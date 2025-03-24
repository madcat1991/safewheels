"""
Vehicle detection using YOLOv8.
"""
import logging
from ultralytics import YOLO

logger = logging.getLogger(__name__)


class VehicleDetector:
    """
    Detects vehicles in images using YOLOv8.
    """

    # YOLOv8 class names for vehicles
    VEHICLE_CLASSES = ['car', 'motorcycle', 'bus', 'truck']

    def __init__(self):
        """
        Initialize the YOLOv8 vehicle detector.

        Args:
            model_path: Path to the YOLOv8 model.
                        If None, will attempt to use a pre-trained model.
        """
        self.model = None
        self._load_model()

    def _load_model(self):
        """
        Load the YOLOv8 model for vehicle detection.

        Args:
            model_path: Path to YOLOv8 model or None to use pre-trained model
        """
        # Use pre-trained YOLOv8 model
        self.model = YOLO("yolov8n.pt")  # Nano model - smallest and fastest
        logger.info("Loaded pre-trained YOLOv8n model")

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
