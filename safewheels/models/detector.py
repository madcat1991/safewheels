"""
Vehicle detection using YOLOv8.
"""
import logging
import torch
from ultralytics import YOLO

logger = logging.getLogger(__name__)


class VehicleDetector:
    """
    Detects vehicles in images using YOLOv8.
    """

    # YOLOv8 class names for vehicles
    VEHICLE_CLASSES = ['car', 'bus', 'truck']

    def __init__(self, filter_orientation=True, use_gpu=True, device=None, model_precision='fp16'):
        """
        Initialize the YOLOv8 vehicle detector.

        Args:
            filter_orientation: Whether to only keep front & rear views.
            use_gpu: Whether to use GPU acceleration if available
            device: Specific device to use (cuda:0, cuda:1, mps, cpu)
            model_precision: Model precision (fp32, fp16)
        """
        self.model = None
        self.filter_orientation = filter_orientation
        self.use_gpu = use_gpu
        self.device = device
        self.model_precision = model_precision
        self._load_model()

    def _load_model(self):
        """
        Load the YOLOv8 model for vehicle detection with GPU acceleration.
        """
        # Determine device
        if self.device is None:
            if self.use_gpu:
                if torch.cuda.is_available():
                    self.device = 'cuda:0'
                    logger.info(f"Using CUDA GPU: {torch.cuda.get_device_name(0)}")
                elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                    self.device = 'mps'
                    logger.info("Using Apple MPS (Metal Performance Shaders)")
                else:
                    self.device = 'cpu'
                    logger.info("GPU requested but not available. Using CPU.")
            else:
                self.device = 'cpu'
                logger.info("Using CPU as requested")

        # Load model with appropriate precision
        if self.model_precision == 'fp16' and self.device.startswith('cuda'):
            # Half precision for faster inference on supported GPUs
            self.model = YOLO("yolov8n.pt").to(self.device).half()
            logger.info("Loaded YOLOv8s model with FP16 precision")
        else:
            # Standard FP32 precision
            self.model = YOLO("yolov8n.pt").to(self.device)
            logger.info(f"Loaded YOLOv8s model on {self.device} with FP32 precision")

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
        # Run YOLOv8 inference with GPU acceleration
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

                # Ensure valid crop bounds
                height, width = image.shape[:2]
                x = max(0, x)
                y = max(0, y)
                w = min(width - x, w)
                h = min(height - y, h)

                # Optional orientation check
                if self.filter_orientation and w > 0 and h > 0:
                    try:
                        orientation = self._classify_orientation(image[y:y+h, x:x+w])
                        if orientation not in ("front", "rear"):
                            continue
                    except Exception as e:
                        logger.warning(f"Error in orientation classification: {e}")
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
        if crop is None or crop.size == 0:
            return "front"  # Default if crop is invalid

        h, w = crop.shape[:2]
        # Very naive ratio check
        if w > 1.3 * h:
            return "side"
        else:
            # We'll just say 'front' (or 'rear'). If you prefer separate logic or a real model,
            # you can implement it here.
            return "front"
