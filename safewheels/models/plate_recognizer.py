"""
License plate detection and character recognition using YOLOv8 for detection and fast-plate-ocr for text recognition.
Optimized for speed and accuracy with European license plates.
"""
import logging
import torch
import cv2
from ultralytics import YOLO
from fast_plate_ocr import ONNXPlateRecognizer

logger = logging.getLogger(__name__)


class EUPlateRecognizer:
    """
    Detects and recognizes license plates using YOLOv8 for detection and fast-plate-ocr for OCR.
    Optimized for German license plates with fast processing.
    """

    def __init__(self, gpu=True, device=None, model_precision='fp16'):
        """
        Initialize the license plate detector and character recognizer.

        Args:
            gpu: Whether to use GPU for detection and recognition
            device: Specific device to use (cuda:0, cuda:1, mps, cpu)
            model_precision: Model precision (fp32, fp16)
        """
        self.plate_detector = None
        self.ocr_recognizer: ONNXPlateRecognizer = None
        self.use_gpu = gpu
        self.device = device
        self.model_precision = model_precision
        self.model_name = 'european-plates-mobile-vit-v2-model'  # Using the European plate model

        self._load_models()

    def _load_models(self):
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

        # Initialize fast-plate-ocr with GPU setting
        try:
            # Create a recognizer with the European model
            # Determine which device to use for fast-plate-ocr
            ocr_device = "cpu"
            if self.use_gpu and self.device.startswith('cuda'):
                ocr_device = "cuda"

            self.ocr_recognizer = ONNXPlateRecognizer(
                hub_ocr_model=self.model_name,
                device=ocr_device  # Using the 'device' parameter with "cuda" or "cpu"
            )
            logger.info(f"Initialized fast-plate-ocr with model: {self.model_name}, device: {ocr_device}")
        except Exception as e:
            logger.error(f"Failed to initialize fast-plate-ocr: {e}")
            raise

    def _detect_plate(self, vehicle_img, confidence_threshold):
        """
        Detect license plate in a vehicle image using YOLOv8.

        Args:
            vehicle_img: Vehicle image (numpy array)
            confidence_threshold: Minimum confidence for detection

        Returns:
            Tuple of (plate_img, plate_bbox, confidence)
            If no plate is detected, returns (None, None, 0.0)
        """
        if vehicle_img is None or vehicle_img.size == 0:
            return None, None, 0.0

        try:
            # Run YOLOv8 inference for license plate detection
            results = self.plate_detector(vehicle_img, conf=confidence_threshold)

            best_plate = None
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
                        best_plate = (x1, y1, w, h)

            if not best_plate:
                logger.debug("No license plate detected in vehicle image")
                return None, None, 0.0

            # Extract plate image
            x, y, w, h = best_plate
            plate_img = vehicle_img[y:y+h, x:x+w].copy() if w > 0 and h > 0 else None

            # Add padding around the plate to improve OCR
            if plate_img is not None and w > 0 and h > 0:
                # Calculate padding (10% of dimensions)
                pad_x = max(int(w * 0.1), 5)
                pad_y = max(int(h * 0.1), 5)

                # Create padded coordinates (ensuring they stay within image boundaries)
                padded_x = max(0, x - pad_x)
                padded_y = max(0, y - pad_y)
                padded_w = min(width - padded_x, w + 2 * pad_x)
                padded_h = min(height - padded_y, h + 2 * pad_y)

                # Extract padded plate image if padding was actually applied
                if padded_x < x or padded_y < y or padded_w > w or padded_h > h:
                    plate_img = vehicle_img[padded_y:padded_y+padded_h, padded_x:padded_x+padded_w].copy()
                    best_plate = (padded_x, padded_y, padded_w, padded_h)

            logger.debug(f"License plate detected with confidence: {best_confidence:.2f}")
            return plate_img, best_plate, best_confidence

        except Exception as e:
            logger.error(f"Error during license plate detection: {e}")
            return None, None, 0.0

    def _preprocess_plate_for_ocr(self, plate_img):
        """
        Preprocess the license plate image for OCR.
        The fast-plate-ocr library expects a grayscale image (H, W) or (H, W, 1).

        Args:
            plate_img: Cropped license plate image

        Returns:
            Preprocessed grayscale plate image
        """
        # Convert to grayscale if it's a color image
        if len(plate_img.shape) == 3 and plate_img.shape[2] == 3:
            gray_plate = cv2.cvtColor(plate_img, cv2.COLOR_BGR2GRAY)
        else:
            gray_plate = plate_img

        return gray_plate

    def _recognize_characters(self, plate_img, confidence_threshold=0.3):
        """
        Recognize characters on the license plate using fast-plate-ocr.

        Args:
            plate_img: Cropped license plate image
            confidence_threshold: Minimum confidence for character recognition

        Returns:
            Tuple of (plate_number, confidence)
            If no characters are recognized, returns (None, 0.0)
        """
        # Using version >= 0.3.0 which provides confidence scores
        try:
            # Preprocess the plate image for OCR
            preprocessed_img = self._preprocess_plate_for_ocr(plate_img)
            if preprocessed_img is None:
                return None, 0.0

            # Run with confidence information
            results = self.ocr_recognizer.run(preprocessed_img, return_confidence=True)

            # Extract text and confidence values
            plate_text = results[0][0]
            confidence = results[1].mean()

            logger.debug(f"OCR result: {plate_text} with confidence {confidence:.2f}")

            # Skip if confidence is too low
            if confidence < confidence_threshold:
                logger.debug(f"OCR confidence too low: {confidence:.2f} < {confidence_threshold}")
                return None, 0.0

        except Exception as e:
            logger.error(f"fast-plate-ocr recognition error: {e}")
            return None, 0.0

        return plate_text, confidence

    def recognize(self, vehicle_img, confidence_threshold=0.4, ocr_confidence_threshold=0.3):
        """
        Detect and recognize license plate in a vehicle image, returning both the plate image and its bounding box.

        Args:
            vehicle_img: Vehicle image (numpy array)
            confidence_threshold: Minimum confidence for plate detection
            ocr_confidence_threshold: Minimum confidence for character recognition

        Returns:
            Tuple of (plate_img, plate_bbox, plate_confidence, plate_number, ocr_confidence)
            If no plate is detected, returns (None, None, 0.0, None, 0.0)
        """
        try:
            # Detect license plate
            plate_img, plate_bbox, plate_confidence = self._detect_plate(
                vehicle_img, confidence_threshold)

            if plate_img is None or plate_bbox is None:
                return None, None, 0.0, None, 0.0

            # Recognize characters on the plate
            plate_number, ocr_confidence = self._recognize_characters(
                plate_img, ocr_confidence_threshold)

            if plate_number:
                # Log the successful detection and recognition
                logger.info(
                    f"Recognized license plate: {plate_number} "
                    f"(plate detection: {plate_confidence:.2f}, OCR: {ocr_confidence:.2f})"
                )
            else:
                # If characters weren't recognized
                plate_number = None
                ocr_confidence = 0.0
                logger.debug(f"License plate detected (conf: {plate_confidence:.2f}) but characters not recognized")

            return plate_img, plate_bbox, plate_confidence, plate_number, ocr_confidence

        except Exception as e:
            logger.error(f"Error during license plate recognition: {e}")
            return None, None, 0.0, None, 0.0
