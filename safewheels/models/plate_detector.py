"""
License plate detection using YOLOv8 for detection.
"""
import logging
import torch
from ultralytics import YOLO

logger = logging.getLogger(__name__)


class PlateDetector:
    """
    Detects license plates using YOLOv8.
    """

    def __init__(self, gpu=True, device=None, model_precision='fp16'):
        """
        Initialize the license plate detector

        Args:
            gpu: Whether to use GPU for detection and recognition
            device: Specific device to use (cuda:0, cuda:1, mps, cpu)
            model_precision: Model precision (fp32, fp16)
        """
        self.plate_detector = None
        self.use_gpu = gpu
        self.device = device
        self.model_precision = model_precision

        self._load_model()

    def _load_model(self):
        """
        Load the YOLOv8 model for license plate detection and initialize fast-plate-ocr.
        """
        # Determine device if not specified
        if self.device is None:
            if self.use_gpu:
                if torch.cuda.is_available():
                    self.device = 'cuda:0'
                    logger.info(f"Using CUDA GPU for plate detection: {torch.cuda.get_device_name(0)}")
                elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                    self.device = 'mps'
                    logger.info("Using Apple MPS for plate detection")
                else:
                    self.device = 'cpu'
                    logger.info("GPU requested but not available. Using CPU for plate detection.")
            else:
                self.device = 'cpu'
                logger.info("Using CPU for plate detection as requested")

        # Load YOLOv8 for license plate detection
        try:
            # Load with appropriate precision based on device and settings
            if self.model_precision == 'fp16' and self.device.startswith('cuda'):
                self.plate_detector = YOLO("yolov8n-lpr.pt").to(self.device).half()
                logger.info("Loaded YOLOv8 LPR model with FP16 precision")
            else:
                self.plate_detector = YOLO("yolov8n-lpr.pt").to(self.device)
                logger.info(f"Loaded YOLOv8 LPR model on {self.device}")
        except Exception as e:
            logger.error(f"Failed to load license plate detection model: {e}")
            raise

    def detect(self, vehicle_img, confidence_threshold):
        """
        Detect license plate in a vehicle image using YOLOv8.

        Args:
            vehicle_img: Vehicle image (numpy array)
            confidence_threshold: Minimum confidence for detection

        Returns:
            Tuple of (plate_bbox, confidence)
            If no plate is detected, returns (None, None, 0.0)
        """
        if vehicle_img is None or vehicle_img.size == 0:
            return None, 0.0

        try:
            # Run YOLOv8 inference for license plate detection
            results = self.plate_detector(vehicle_img, conf=confidence_threshold)

            best_plate_bbox = None
            best_confidence = 0.0

            # Process detection results
            for result in results:
                boxes = result.boxes

                for box in boxes:
                    # Get bounding box coordinates and confidence
                    x1, y1, x2, y2 = box.xyxy[0].tolist()
                    confidence = box.conf.item()

                    # Convert to integer coordinates
                    x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)

                    # Ensure coordinates are within image boundaries
                    height, width = vehicle_img.shape[:2]
                    x1 = max(0, x1)
                    y1 = max(0, y1)
                    x2 = min(width, x2)
                    y2 = min(height, y2)

                    # Calculate width and height
                    w = x2 - x1
                    h = y2 - y1

                    # Skip if resulting area is too small
                    if w < 20 or h < 10:
                        continue

                    # Update best plate if confidence is higher
                    if confidence > best_confidence:
                        best_confidence = confidence
                        best_plate_bbox = (x1, y1, w, h)

            if not best_plate_bbox:
                logger.debug("No license plate detected in vehicle image")
                return None, 0.0

            logger.debug(f"License plate detected with confidence: {best_confidence:.2f}")
            return best_plate_bbox, best_confidence

        except Exception as e:
            logger.error(f"Error during license plate detection: {e}")
            return None, 0.0
