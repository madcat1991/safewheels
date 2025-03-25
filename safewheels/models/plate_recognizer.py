"""
License plate detection and character recognition using YOLOv8 for detection and EasyOCR for text recognition.
"""
import logging
import cv2
from ultralytics import YOLO
import easyocr

logger = logging.getLogger(__name__)


class PlateRecognizer:
    """
    Detects and recognizes license plates using YOLOv8 for detection and EasyOCR for text recognition.
    """

    def __init__(self, languages=None, gpu=True):
        """
        Initialize the license plate detector and character recognizer.

        Args:
            languages: List of language codes for EasyOCR (default: ['en', 'de', 'uk'])
            gpu: Whether to use GPU for EasyOCR (default: True)
        """
        self.plate_detector = None
        self.ocr_reader = None
        self.languages = languages or ['en', 'de', 'uk']
        self.use_gpu = gpu
        self._load_models()

    def _load_models(self):
        """
        Load the YOLOv8 model for license plate detection and EasyOCR for text recognition.
        """
        # Load YOLOv8 for license plate detection
        try:
            self.plate_detector = YOLO("yolov8n-lpr.pt")
            logger.info("Loaded pre-trained YOLOv8 model for license plate detection")
        except Exception as e:
            logger.error(f"Failed to load license plate detection model: {e}")
            raise

        # Initialize EasyOCR with specified language support
        try:
            self.ocr_reader = easyocr.Reader(self.languages, gpu=self.use_gpu)
            logger.info(f"Initialized EasyOCR with language support for: {', '.join(self.languages)}")
        except Exception as e:
            logger.error(f"Failed to initialize EasyOCR with languages {self.languages}: {e}")
            raise

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

        try:
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
                logger.debug("No license plate detected in vehicle image")
                return None, None, 0.0

            # Extract plate image
            x, y, w, h = best_plate
            plate_img = vehicle_img[y:y+h, x:x+w].copy() if w > 0 and h > 0 else None

            # Add a small padding around the plate (if possible) to improve OCR
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

    def recognize_characters(self, plate_img, confidence_threshold=0.3):
        """
        Recognize characters on the license plate using EasyOCR.

        Args:
            plate_img: Cropped license plate image
            confidence_threshold: Minimum confidence for character recognition

        Returns:
            Tuple of (plate_number, confidence)
            If no characters are recognized, returns (None, 0.0)
        """
        if plate_img is None or plate_img.size == 0:
            return None, 0.0

        try:
            # Preprocess the plate image for OCR
            plate_preprocessed = self._preprocess_plate(plate_img)

            # Run EasyOCR on the preprocessed plate image
            results = self.ocr_reader.readtext(plate_preprocessed)

            # If no text detected
            if not results:
                logger.debug("No text detected in license plate image")
                return None, 0.0

            # Filter results by confidence threshold
            valid_results = [result for result in results if result[2] >= confidence_threshold]

            if not valid_results:
                logger.debug(f"No text with confidence >= {confidence_threshold} detected")
                return None, 0.0

            # Sort results by x-coordinate to ensure correct reading order
            valid_results.sort(key=lambda x: x[0][0][0])  # Sort by x-coordinate of first point

            # Extract text and confidences
            texts = [result[1] for result in valid_results]
            confidences = [result[2] for result in valid_results]

            # Join texts (with space between multiple detections)
            plate_text = " ".join(texts)

            # Calculate average confidence
            avg_confidence = sum(confidences) / len(confidences)

            # Clean up the recognized text (remove spaces, control characters, etc.)
            plate_number = ''.join(c for c in plate_text if c.isalnum())

            logger.debug(f"Recognized plate: {plate_number} (confidence: {avg_confidence:.2f})")
            return plate_number, avg_confidence

        except Exception as e:
            logger.error(f"Error during character recognition: {e}")
            return None, 0.0

    def _preprocess_plate(self, plate_img):
        """
        Preprocess the license plate image for better character recognition with EasyOCR.

        Args:
            plate_img: Cropped license plate image

        Returns:
            Preprocessed image ready for OCR
        """
        if plate_img is None:
            return None

        try:
            # Resize to appropriate size for OCR while maintaining aspect ratio
            height, width = plate_img.shape[:2]
            new_width = 300  # Standard width that works well with EasyOCR
            new_height = int(height * (new_width / width))
            resized = cv2.resize(plate_img, (new_width, new_height), interpolation=cv2.INTER_AREA)

            # Convert to grayscale
            gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY) if len(resized.shape) > 2 else resized

            # Apply CLAHE (Contrast Limited Adaptive Histogram Equalization)
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            enhanced = clahe.apply(gray)

            # Apply bilateral filter to reduce noise while preserving edges
            processed = cv2.bilateralFilter(enhanced, 11, 17, 17)

            # Optionally apply additional processing for clearer text
            # Apply mild sharpening using unsharp masking
            blurred = cv2.GaussianBlur(processed, (0, 0), 3)
            sharpened = cv2.addWeighted(processed, 1.5, blurred, -0.5, 0)

            return sharpened

        except Exception as e:
            logger.error(f"Error during image preprocessing: {e}")
            # Return original image if preprocessing fails
            return plate_img

    def recognize(self, vehicle_img, confidence_threshold=0.4, ocr_confidence_threshold=0.3):
        """
        Detect and recognize license plate in a vehicle image.

        Args:
            vehicle_img: Vehicle image (numpy array)
            confidence_threshold: Minimum confidence for plate detection
            ocr_confidence_threshold: Minimum confidence for character recognition

        Returns:
            Tuple of (plate_img, plate_number, confidence)
            If no plate is detected or recognized, returns (None, None, 0.0)
        """
        try:
            # Detect license plate
            plate_img, plate_bbox, plate_confidence = self.detect_plate(
                vehicle_img, confidence_threshold)

            if plate_img is None:
                return None, None, 0.0

            # Recognize characters on the plate
            plate_number, ocr_confidence = self.recognize_characters(
                plate_img, ocr_confidence_threshold)

            # Calculate overall confidence
            if plate_number:
                # Combine plate detection and OCR confidences (weighted average)
                confidence = 0.4 * plate_confidence + 0.6 * ocr_confidence

                # Log the successful detection and recognition
                logger.info(f"Recognized license plate: {plate_number} with confidence {confidence:.2f}")
            else:
                # If characters weren't recognized, return just the plate confidence
                plate_number = None
                confidence = plate_confidence
                logger.debug("License plate detected but characters not recognized")

            return plate_img, plate_number, confidence

        except Exception as e:
            logger.error(f"Error during license plate recognition: {e}")
            return None, None, 0.0
