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

    def __init__(self, filter_orientation=True):
        """
        Initialize the YOLOv8 vehicle detector.

        Args:
            model_path: Path to the YOLOv8 model.
                        If None, will attempt to use a pre-trained model.
            filter_orientation: Whether to only keep front & rear views.
        """
        self.model = None
        self.filter_orientation = filter_orientation
        self._load_model()

    def _load_model(self):
        """
        Load the YOLOv8 model for vehicle detection.

        Args:
            model_path: Path to YOLOv8 model or None to use pre-trained model
        """
        # Use pre-trained YOLOv8 model
        self.model = YOLO("yolov8s.pt")
        logger.info("Loaded pre-trained YOLOv8s model")

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
            boxes = result.boxes
            names = result.names

            for box in boxes:
                cls_id = int(box.cls.item())
                cls_name = names[cls_id]

                # Filter for vehicle classes only
                if cls_name not in self.VEHICLE_CLASSES:
                    continue

                # Get bounding box coords
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                x = int(x1)
                y = int(y1)
                w = int(x2 - x1)
                h = int(y2 - y1)

                # Optional orientation check
                if self.filter_orientation:
                    orientation = self._classify_orientation(image[y:y+h, x:x+w])
                    if orientation not in ("front", "rear"):
                        continue

                confidence = float(box.conf.item())

                vehicles.append({
                    'bbox': [x, y, w, h],
                    'confidence': confidence,
                    'class_name': cls_name
                })

        return vehicles

    def _classify_orientation(self, crop):
        """
        Naive orientation classifier stub. Returns one of: 'front', 'rear', 'side'.
        For demonstration, we simply check aspect ratio:
            - If the bounding box is significantly wider than tall, treat it as 'side'.
            - Otherwise, treat it as 'front'. (We do not differentiate front vs. rear here.
              But you could upgrade this with a real orientation classifier or heuristic.)
        """
        h, w = crop.shape[:2]
        # Very naive ratio check
        if w > 1.3 * h:
            return "side"
        else:
            # We'll just say 'front' (or 'rear'). If you prefer separate logic or a real model,
            # you can implement it here.
            return "front"
