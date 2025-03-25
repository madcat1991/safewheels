"""
License plate detection and character recognition using YOLOv8 for detection and EasyOCR for OCR.
Optimized for speed and accuracy with European license plates (primarily English and German).
"""
import logging
import cv2
import re
from ultralytics import YOLO
import easyocr

logger = logging.getLogger(__name__)


class PlateRecognizer:
    """
    Detects and recognizes license plates using YOLOv8 for detection and EasyOCR for OCR.
    Optimized for English and German license plates with fast processing.
    """

    def __init__(self, languages=None, gpu=True):
        """
        Initialize the license plate detector and character recognizer.

        Args:
            languages: List of language codes for EasyOCR (default: ['en', 'de'])
            gpu: Whether to use GPU for detection and recognition
        """
        self.plate_detector = None
        self.reader = None
        self.use_gpu = gpu
        self.languages = languages or ['en', 'de']  # English and German only for faster processing
        self._load_models()

        # Common license plate patterns (can be extended)
        self.plate_patterns = {
            'european': r'[A-ZÄÖÜ]{1,3}[-\s]?[0-9]{1,4}[-\s]?[A-ZÄÖÜß0-9]{1,3}',
            'german': r'[A-ZÄÖÜ]{1,3}[-\s]?[A-Z]{1,2}[-\s]?[0-9]{1,4}',  # Standard German format
            'generic': r'[A-ZÄÖÜ0-9]{3,8}'  # Generic alphanumeric pattern
        }

    def _load_models(self):
        """
        Load the YOLOv8 model for license plate detection and EasyOCR.
        """
        # Load YOLOv8 for license plate detection
        try:
            self.plate_detector = YOLO("yolov8n-lpr.pt")
            logger.info("Loaded pre-trained YOLOv8 model for license plate detection")
        except Exception as e:
            logger.error(f"Failed to load license plate detection model: {e}")
            raise

        # Initialize EasyOCR reader
        try:
            # Initialize EasyOCR reader with specified languages
            self.reader = easyocr.Reader(
                self.languages,
                gpu=self.use_gpu
            )
            logger.info(f"Initialized EasyOCR with languages: {self.languages}")
        except Exception as e:
            logger.error(f"Failed to initialize EasyOCR: {e}")
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

    def _preprocess_plate(self, plate_img):
        """
        Preprocess the license plate image for better character recognition.

        Args:
            plate_img: Cropped license plate image

        Returns:
            List of preprocessed images ready for OCR
        """
        if plate_img is None:
            return []

        try:
            processed_images = []

            # Resize to appropriate size for OCR while maintaining aspect ratio
            height, width = plate_img.shape[:2]
            new_width = 300  # Standard width that works well with OCR
            new_height = int(height * (new_width / width))
            # AREA for downsampling
            resized = cv2.resize(plate_img, (new_width, new_height), interpolation=cv2.INTER_AREA)
            processed_images.append(resized)

            # Convert to grayscale
            gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY) if len(resized.shape) > 2 else resized

            # Apply CLAHE (Contrast Limited Adaptive Histogram Equalization)
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            enhanced = clahe.apply(gray)
            processed_images.append(enhanced)

            # Apply bilateral filter to reduce noise while preserving edges - faster parameters
            bilateral = cv2.bilateralFilter(enhanced, 5, 50, 50)  # Smaller diameter for faster processing
            processed_images.append(bilateral)

            # Only create one threshold version - use Otsu as it's generally reliable
            _, otsu = cv2.threshold(bilateral, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            processed_images.append(otsu)

            return processed_images

        except Exception as e:
            logger.error(f"Error during image preprocessing: {e}")
            # Return original image if preprocessing fails
            return [plate_img]

    def _clean_plate_text(self, text):
        """
        Clean and normalize license plate text.

        Args:
            text: Raw text from OCR

        Returns:
            Cleaned license plate string
        """
        if not text:
            return None

        # Convert to uppercase
        text = text.upper()

        # Remove spaces, dots, and other non-alphanumeric characters
        text = re.sub(r'[\s\.,;:_\-\'\"\(\)\[\]\{\}]', '', text)

        # Common OCR corrections for license plates
        corrections = {
            '0': 'O',  # Sometimes 0 is mistaken for O in German plates
            '1': 'I',  # 1 can be mistaken for I
            '8': 'B',  # 8 can be mistaken for B
            '5': 'S',  # 5 can be mistaken for S
            '2': 'Z',  # 2 can be mistaken for Z
            # NOTE: We don't automatically convert A->Ä, O->Ö, U->Ü generally
            # as this would cause more problems than it solves.
            # Instead, we handle specific city codes in the special cases below
        }

        # Apply corrections only if the result looks like a plate
        # German plates typically start with 1-3 letters (city code) followed by separators and identifier
        if re.match(r'^[A-ZÄÖÜ0-9]{1,3}[-\s]?', text):
            # This is likely a German plate, apply city code corrections
            first_part_match = re.match(r'^([A-ZÄÖÜ0-9]{1,3})', text)
            if first_part_match:
                first_part = first_part_match.group(1)
                rest_of_text = text[len(first_part):]

                # First handle digit corrections in the city code
                digit_corrections = {'0': 'O', '1': 'I', '8': 'B', '5': 'S', '2': 'Z'}
                for digit, letter in digit_corrections.items():
                    first_part = first_part.replace(digit, letter)

                # Check for common German city codes with umlauts (according to official codes)
                # Format based on https://en.wikipedia.org/wiki/Vehicle_registration_plates_of_Germany
                umlaut_corrections = {
                    'MU': 'MÜ',  # München
                    'LO': 'LÖ',  # Lörrach
                    'TU': 'TÜ',  # Tübingen
                    'FU': 'FÜ',  # Fürth
                    'GO': 'GÖ',  # Göttingen
                    'KO': 'KÖ',  # Köln area districts
                }
                
                # Apply the umlaut correction if this is a known city code
                if first_part in umlaut_corrections:
                    logger.debug(f"Correcting city code: {first_part} -> {umlaut_corrections[first_part]}")
                    first_part = umlaut_corrections[first_part]

                cleaned_text = first_part + rest_of_text
            else:
                cleaned_text = text
        else:
            cleaned_text = text

        # Keep only alphanumeric characters including German umlauts (based on official German license plate standard)
        allowed_chars = set('ABCDEFGHIJKLMNOPQRSTUVWXYZÄÖÜ0123456789')
        cleaned_text = ''.join(c for c in cleaned_text if c in allowed_chars)

        # Check minimum length to be a valid plate
        if len(cleaned_text) < 3:
            return None

        return cleaned_text

    def _is_valid_plate(self, plate_text):
        """
        Validate if the text looks like a license plate.

        Args:
            plate_text: Cleaned plate text

        Returns:
            Boolean indicating if text pattern matches a license plate
        """
        if not plate_text or len(plate_text) < 3:
            return False

        # Check against known patterns
        for pattern in self.plate_patterns.values():
            if re.match(pattern, plate_text):
                return True

        # General pattern: Check if the text contains at least some letters and digits
        letters = sum(1 for c in plate_text if c.isalpha())
        digits = sum(1 for c in plate_text if c.isdigit())

        # Most license plates have at least 1 letter and 1 digit
        if letters >= 1 and digits >= 1 and 4 <= len(plate_text) <= 8:
            return True

        # Special case: Full numeric plates (rare but exist)
        if digits >= 3 and len(plate_text) >= 3 and len(plate_text) <= 8:
            return True

        return False

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
            processed_images = self._preprocess_plate(plate_img)
            if not processed_images:
                return None, 0.0

            all_results = []

            # Recognize text in each processed image using two different methods
            for img in processed_images:
                try:
                    # Convert to RGB for EasyOCR if needed
                    if len(img.shape) == 2:  # If grayscale
                        img_for_ocr = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
                    else:
                        img_for_ocr = img

                    # Fast method with greedy decoder only
                    detection_results1 = self.reader.readtext(
                        img_for_ocr,
                        detail=1,  # Return bounding boxes and confidences
                        paragraph=False,  # Treat each text box separately
                        decoder='greedy',  # Greedy decoder is much faster
                        allowlist='ABCDEFGHIJKLMNOPQRSTUVWXYZÄÖÜ0123456789',  # Standard Latin and German license plate characters
                        batch_size=1,  # Process one image at a time
                        mag_ratio=1.5,  # Moderate magnification ratio for speed
                        canvas_size=1280,  # Smaller canvas size for faster processing
                        contrast_ths=0.2,  # Lower contrast threshold for low contrast plates
                        adjust_contrast=1.3  # Moderate contrast adjustment
                    )

                    # We only use the greedy decoder for faster processing

                    # Process results from greedy decoder only (for speed)
                    for box, text, confidence in detection_results1:
                        if confidence >= confidence_threshold:
                            # Skip very short segments that are likely part of another detection
                            if len(text) <= 2 and any(r[0].startswith(text) or r[0].endswith(text) for r in all_results):
                                continue

                            cleaned_text = self._clean_plate_text(text)
                            if cleaned_text:
                                all_results.append((cleaned_text, confidence, "easyocr"))

                except Exception as e:
                    logger.debug(f"EasyOCR error on image: {e}")

            # Filter by confidence threshold
            valid_results = [r for r in all_results if r[1] >= confidence_threshold]

            if not valid_results:
                logger.debug(f"No text with confidence >= {confidence_threshold} detected")
                return None, 0.0

            # Order by confidence
            valid_results.sort(key=lambda x: x[1], reverse=True)

            # Find the first valid license plate format
            for text, conf, method in valid_results:
                if self._is_valid_plate(text):
                    logger.debug(f"Valid plate detected: {text} (conf: {conf:.2f}, method: {method})")
                    return text, conf

            # If no valid plates, return the highest confidence result
            best_result = valid_results[0]
            logger.debug(
                f"Best plate candidate: {best_result[0]} (conf: {best_result[1]:.2f}, method: {best_result[2]})"
            )
            return best_result[0], best_result[1]

        except Exception as e:
            logger.error(f"Error during character recognition: {e}")
            return None, 0.0

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
