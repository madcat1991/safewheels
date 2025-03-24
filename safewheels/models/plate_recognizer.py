"""
License plate detection and character recognition using pre-trained YOLOv8 models.
"""
import logging
import cv2
from ultralytics import YOLO

logger = logging.getLogger(__name__)


class PlateRecognizer:
    """
    Detects and recognizes license plates using pre-trained YOLOv8 models.
    """

    def __init__(self):
        """
        Initialize the license plate detector and character recognizer.
        """
        self.plate_detector = None
        self.ocr_model = None
        self._load_models()

    def _load_models(self):
        """
        Load the pre-trained YOLOv8 models for license plate detection and character recognition.
        """
        # Load YOLOv8 for license plate detection
        self.plate_detector = YOLO("yolov8n-lpr.pt")
        logger.info("Loaded pre-trained YOLOv8 model for license plate detection")

        # Load YOLOv8 for OCR
        self.ocr_model = YOLO("yolov8n-ocr.pt")
        logger.info("Loaded pre-trained YOLOv8 model for license plate OCR")

    def detect_plate(self, vehicle_img, confidence_threshold=0.4):
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

        # Run YOLOv8 inference for license plate detection
        results = self.plate_detector(vehicle_img, conf=confidence_threshold)

        best_plate = None
        best_confidence = 0.0

        # Process detection results
        for result in results:
            boxes = result.boxes

            for box in boxes:
                # Get bounding box coordinates (x1, y1, x2, y2)
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
            return None, None, 0.0

        # Extract plate image
        x, y, w, h = best_plate
        plate_img = vehicle_img[y:y+h, x:x+w].copy() if w > 0 and h > 0 else None

        return plate_img, best_plate, best_confidence

    def recognize_characters(self, plate_img, confidence_threshold=0.3):
        """
        Recognize characters on the license plate using YOLOv8 OCR model.

        Args:
            plate_img: Cropped license plate image
            confidence_threshold: Minimum confidence for character recognition

        Returns:
            Tuple of (plate_number, confidence)
            If no characters are recognized, returns (None, 0.0)
        """
        if plate_img is None or plate_img.size == 0:
            return None, 0.0

        # Preprocess the plate image for OCR
        plate_preprocessed = self._preprocess_plate(plate_img)

        # Run YOLOv8 inference for character recognition
        results = self.ocr_model(plate_preprocessed, conf=confidence_threshold)

        char_detections = []

        # Process OCR results
        for result in results:
            boxes = result.boxes

            for box in boxes:
                # Get bounding box coordinates (x1, y1, x2, y2)
                x1 = box.xyxy[0][0].item()
                conf = box.conf.item()
                cls_id = int(box.cls.item())

                # Get character from class name
                char = result.names[cls_id]

                # Add to detections with x-coordinate for sorting
                char_detections.append((x1, char, conf))

        # If no characters detected
        if not char_detections:
            return None, 0.0

        # Sort by x-coordinate (left to right reading order)
        char_detections.sort(key=lambda x: x[0])

        # Extract characters and confidences
        plate_chars = [det[1] for det in char_detections]
        confidences = [det[2] for det in char_detections]

        # Join characters to form the plate number
        plate_number = ''.join(plate_chars)

        # Calculate average confidence
        avg_confidence = sum(confidences) / len(confidences)

        return plate_number, avg_confidence

    def _preprocess_plate(self, plate_img):
        """
        Preprocess the license plate image for better character recognition.

        Args:
            plate_img: Cropped license plate image

        Returns:
            Preprocessed image ready for OCR
        """
        if plate_img is None:
            return None

        # Resize to appropriate size for OCR
        resized = cv2.resize(plate_img, (224, 64), interpolation=cv2.INTER_AREA)

        # Convert to grayscale
        gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY) if len(resized.shape) > 2 else resized

        # Apply adaptive thresholding
        thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                       cv2.THRESH_BINARY, 11, 2)

        # Apply Gaussian blur to reduce noise
        blurred = cv2.GaussianBlur(thresh, (5, 5), 0)

        # Convert back to 3 channels for model input
        processed = cv2.cvtColor(blurred, cv2.COLOR_GRAY2BGR)

        return processed

    def recognize(self, vehicle_img, confidence_threshold=0.4):
        """
        Detect and recognize license plate in a vehicle image.

        Args:
            vehicle_img: Vehicle image (numpy array)
            confidence_threshold: Minimum confidence for detection and recognition

        Returns:
            Tuple of (plate_img, plate_number, confidence)
            If no plate is detected or recognized, returns (None, None, 0.0)
        """
        # Detect license plate
        plate_img, plate_bbox, plate_confidence = self.detect_plate(
            vehicle_img, confidence_threshold)

        if plate_img is None:
            return None, None, 0.0

        # Recognize characters on the plate
        plate_number, ocr_confidence = self.recognize_characters(
            plate_img, confidence_threshold)

        # Calculate overall confidence
        if plate_number:
            # Combine plate detection and OCR confidences
            confidence = min(plate_confidence, ocr_confidence)
        else:
            # If characters weren't recognized, return just the plate confidence
            plate_number = None
            confidence = plate_confidence

        return plate_img, plate_number, confidence
