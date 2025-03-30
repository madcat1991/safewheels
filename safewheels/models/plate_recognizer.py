"""
License plate detection and character recognition using YOLOv8 for detection and EasyOCR for OCR.
Optimized for speed and accuracy with European license plates (primarily English and German).
"""
import logging
import math
import cv2
import re
import numpy as np
from ultralytics import YOLO
import easyocr

logger = logging.getLogger(__name__)


MIN_PLATE_LEN = 2


def rotate_image(image, angle):
    """Rotate an image around its center by the given angle.

    Args:
        image: The input image (numpy array)
        angle: The rotation angle in degrees (positive = counterclockwise)

    Returns:
        The rotated image
    """
    if image is None or image.size == 0:
        return image

    # Calculate image center
    image_center = tuple(np.array(image.shape[1::-1]) / 2)

    # Calculate new image dimensions to avoid cropping
    height, width = image.shape[:2]
    cos_angle = abs(np.cos(np.radians(angle)))
    sin_angle = abs(np.sin(np.radians(angle)))
    new_width = int(width * cos_angle + height * sin_angle)
    new_height = int(height * cos_angle + width * sin_angle)

    # Get rotation matrix
    rot_mat = cv2.getRotationMatrix2D(image_center, angle, 1.0)

    # Adjust the rotation matrix to account for new dimensions
    rot_mat[0, 2] += (new_width - width) / 2
    rot_mat[1, 2] += (new_height - height) / 2

    # Perform the rotation with border replication to avoid artifacts
    result = cv2.warpAffine(
        image,
        rot_mat,
        (new_width, new_height),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REPLICATE
    )

    return result


def compute_skew(src_img, max_angle_deg=30):
    """Compute the skew angle of text in an image.

    Args:
        src_img: Source image (numpy array)
        max_angle_deg: Maximum angle to consider as valid skew (degrees)

    Returns:
        Estimated skew angle in degrees
    """
    if src_img is None or src_img.size == 0:
        return 0.0

    try:
        # Determine image dimensions
        if len(src_img.shape) == 3:
            h, w, _ = src_img.shape
        elif len(src_img.shape) == 2:
            h, w = src_img.shape
        else:
            logger.warning("Unsupported image type for skew detection")
            return 0.0

        # Ensure minimum size for processing
        if h < 20 or w < 20:
            return 0.0

        # Convert to grayscale if needed
        if len(src_img.shape) == 3:
            gray = cv2.cvtColor(src_img, cv2.COLOR_BGR2GRAY)
        else:
            gray = src_img.copy()

        # Apply median blur to reduce noise while preserving edges
        img = cv2.medianBlur(gray, 3)

        # Binarize the image to enhance edge detection
        _, binary = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

        # Detect edges using Canny
        edges = cv2.Canny(binary, threshold1=30, threshold2=100, apertureSize=3, L2gradient=True)

        # Find lines using probabilistic Hough transform
        # Adjust parameters based on image dimensions
        min_line_length = max(w / 6.0, 20)  # Min line length should be proportional to image width
        max_line_gap = max(h / 6.0, 10)     # Max gap between line segments

        lines = cv2.HoughLinesP(
            edges,
            rho=1,
            theta=math.pi/180,
            threshold=max(30, min(h, w) // 10),  # Adaptive threshold based on image size
            minLineLength=min_line_length,
            maxLineGap=max_line_gap
        )

        if lines is None or len(lines) == 0:
            return 0.0

        # Process detected lines to compute skew
        angles = []
        max_angle_rad = math.radians(max_angle_deg)

        for line in lines:
            for x1, y1, x2, y2 in line:
                # Skip vertical and near-vertical lines
                if abs(x2 - x1) < 5:
                    continue

                # Calculate angle
                ang = np.arctan2(y2 - y1, x2 - x1)

                # Normalize angle to be between -pi/2 and pi/2
                while ang < -math.pi/2:
                    ang += math.pi
                while ang > math.pi/2:
                    ang -= math.pi

                # Only consider angles within max_angle_rad
                if abs(ang) <= max_angle_rad:
                    angles.append(ang)

        if len(angles) == 0:
            return 0.0

        # Use median angle to reduce impact of outliers
        median_angle = np.median(angles)
        return math.degrees(median_angle)

    except Exception as e:
        logger.error(f"Error in skew detection: {e}")
        return 0.0


def deskew(src_img, max_angle_deg=30):
    """Correct the skew in an image by rotating it to align text horizontally.

    Args:
        src_img: Source image (numpy array)
        max_angle_deg: Maximum angle to consider as valid skew (degrees)

    Returns:
        Deskewed image
    """
    if src_img is None or src_img.size == 0:
        return src_img

    try:
        # Compute skew angle
        angle = compute_skew(src_img, max_angle_deg)

        # Skip rotation if angle is very small (reduces unnecessary processing)
        if abs(angle) < 0.5:
            return src_img

        # Apply rotation to correct skew
        deskewed_img = rotate_image(src_img, angle)

        # Ensure we didn't create a much larger image
        orig_size = src_img.size
        new_size = deskewed_img.size

        # If new image is more than 50% larger, revert to original
        if new_size > orig_size * 1.5:
            logger.warning("Deskewed image size increased significantly, using original")
            return src_img

        return deskewed_img
    except Exception as e:
        logger.error(f"Error in deskew: {e}")
        return src_img


class DePlateRecognizer:
    """
    Detects and recognizes license plates using YOLOv8 for detection and EasyOCR for OCR.
    Optimized for German license plates with fast processing.
    """

    def __init__(self, gpu=True):
        """
        Initialize the license plate detector and character recognizer.

        Args:
            gpu: Whether to use GPU for detection and recognition
        """
        self.plate_detector = None
        self.reader = None
        self.use_gpu = gpu
        self.languages = ['de']

        # Character confusion maps for text correction
        self.digit_to_letter = {'0': 'O', '1': 'I', '8': 'B', '5': 'S', '2': 'Z'}
        self.letter_to_digit = {'O': '0', 'I': '1', 'B': '8', 'S': '5', 'Z': '2'}

        # Common German city codes with umlauts
        self.umlaut_corrections = {
            "AO": "AÖ",  # Altötting
            "BUD": "BÜD",  # Büdingen
            "BUR": "BÜR",  # Büren
            "BUS": "BÜS",  # Büsingen
            "BUZ": "BÜZ",  # Bützow
            "DUW": "DÜW",  # Bad Dürkheim an der Weinstraße
            "FLO": "FLÖ",  # Flöha
            "FU": "FÜ",  # Fürth
            "FUS": "FÜS",  # Füssen
            "GU": "GÜ",  # Güstrow
            "HMU": "HÜM",  # Hann. Münden
            "HOS": "HÖS",  # Höchstadt
            "JUL": "JÜL",  # Jülich
            "KON": "KÖN",  # Bad Königshofen
            "KOT": "KÖT",  # Köthen
            "KOZ": "KÖZ",  # Bad Kötzting
            "KUN": "KÜN",  # Künzelsau
            "LO": "LÖ",  # Lörrach
            "LOB": "LÖB",  # Löbau
            "LUN": "LÜN",  # Lünen
            "MU": "MÜ",  # Mühldorf
            "MUB": "MÜB",  # Münchberg
            "MUR": "MÜR",  # Müritz
            "NO": "NÖ",  # Nördlingen
            "PLO": "PLÖ",  # Plön
            "PRU": "PRÜ",  # Prüm
            "RUD": "RÜD",  # Rüdesheim
            "RUG": "RÜG",  # Rügen
            "SAK": "SÄK",  # Bad Säckingen
            "SLU": "SLÜ",  # Schlüchtern
            "SMU": "SMÜ",  # Schwabmünchen
            "SOM": "SÖM",  # Sömmerda
            "SUW": "SÜW",  # Südliche Weinstraße
            "TOL": "TÖL",  # Bad Tölz
            "TU": "TÜ",  # Tübingen
            "UB": "ÜB",  # Überlingen
            "WU": "WÜ",  # Würzburg
            "WUM": "WÜM",  # Waldmünchen
        }

        # Common license plate patterns
        self.plate_pattern = r'^(?:[A-ZÄÖÜ]{1,3}[-\s]?[A-Z]{1,2}[-\s]?\d{1,4}[HE]?|[0,1]-\d{1,5}|Y-\d{1,6})$'

        self._load_models()

    def _load_models(self):
        """
        Load the YOLOv8 model for license plate detection and initialize EasyOCR.
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
            self.reader = easyocr.Reader(
                self.languages,
                gpu=self.use_gpu
            )
            logger.info(f"Initialized EasyOCR with languages: {self.languages}")
        except Exception as e:
            logger.error(f"Failed to initialize EasyOCR: {e}")
            raise

    def detect_plate(self, vehicle_img, confidence_threshold):
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

    def _preprocess_plate(self, plate_img):
        """
        Preprocess the license plate image for better character recognition.
        Optimized with multiple processing steps to improve OCR accuracy.

        Args:
            plate_img: Cropped license plate image

        Returns:
            Dictionary of preprocessed images with processing type as key
        """
        if plate_img is None:
            return {}

        try:
            processed_images = {}

            # Resize to appropriate size for OCR while maintaining aspect ratio
            height, width = plate_img.shape[:2]

            # Enhanced resizing for better OCR results
            new_width = 400  # Optimal width for OCR
            new_height = int(height * (new_width / width))

            # Ensure the height is at least 50px for better OCR results
            if new_height < 50:
                scale_factor = 50 / new_height
                new_height = 50
                new_width = int(new_width * scale_factor)

            # Use INTER_CUBIC for upsampling to get sharper text edges
            resized = cv2.resize(
                plate_img,
                (new_width, new_height),
                interpolation=cv2.INTER_CUBIC
            )
            processed_images['resized'] = resized

            # Convert to grayscale
            gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY) if len(resized.shape) > 2 else resized
            processed_images['gray'] = gray

            # Apply CLAHE for better contrast
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            enhanced = clahe.apply(gray)
            processed_images['enhanced'] = enhanced

            # Apply Gaussian blur to remove noise
            blurred = cv2.GaussianBlur(enhanced, (3, 3), 0)
            processed_images['blurred'] = blurred

            # Otsu's thresholding for optimal binarization
            _, otsu = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            processed_images['binary'] = otsu

            # Add adaptive thresholding which often works better for uneven lighting
            adaptive = cv2.adaptiveThreshold(
                blurred,
                255,
                cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                cv2.THRESH_BINARY,
                11,
                2
            )
            processed_images['adaptive'] = adaptive

            return processed_images

        except Exception as e:
            logger.error(f"Error during image preprocessing: {e}")
            # Return original image if preprocessing fails
            if plate_img is not None and len(plate_img.shape) >= 2:
                return {'original': plate_img}
            return {}

    def _clean_plate_text(self, text):
        """
        Clean and normalize license plate text with improved character correction.

        Args:
            text: Raw text from OCR

        Returns:
            Cleaned license plate string
        """
        if not text:
            return None

        # Convert to uppercase
        text = text.upper()

        # Apply corrections only if the result looks like a plate
        # German plates typically start with 1-3 letters (city code) followed by separators and identifier
        if re.match(r'^[A-ZÄÖÜ0-9]{1,3}[-\s]?', text):
            # This is likely a German plate, apply city code corrections
            first_part_match = re.match(r'^([A-ZÄÖÜ0-9]{1,3})', text)
            if first_part_match:
                first_part = first_part_match.group(1)
                rest_of_text = text[len(first_part):]

                # First handle digit corrections in the city code (except for '0' and '1')
                if first_part not in ('0', '1'):
                    for digit, letter in self.digit_to_letter.items():
                        first_part = first_part.replace(digit, letter)

                # Apply the umlaut correction if this is a known city code
                if first_part in self.umlaut_corrections:
                    logger.debug(f"Correcting city code: {first_part} -> {self.umlaut_corrections[first_part]}")
                    first_part = self.umlaut_corrections[first_part]

                # If there's a dash followed by digits at the end, apply specific corrections
                num_part_match = re.search(r'-([A-Z0-9]+)$', rest_of_text)
                if num_part_match:
                    num_part = num_part_match.group(1)
                    num_part_start = rest_of_text.rfind('-' + num_part)

                    # Apply opposite corrections in the numeric part
                    # In this part, change 'O' to '0', 'I' to '1', etc.
                    corrected_num = num_part
                    for letter, digit in self.letter_to_digit.items():
                        corrected_num = corrected_num.replace(letter, digit)

                    rest_of_text = rest_of_text[:num_part_start+1] + corrected_num

                cleaned_text = first_part + rest_of_text
            else:
                cleaned_text = text
        else:
            cleaned_text = text

        # Check minimum length to be a valid plate
        if len(cleaned_text) < MIN_PLATE_LEN:
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
        if not plate_text or len(plate_text) < MIN_PLATE_LEN:
            return False

        # Check against known patterns
        if re.match(self.plate_pattern, plate_text):
            return True

        return False

    def _run_ocr_on_image(self, img, allowed_chars, use_beam_search=False):
        """
        Run OCR on a single preprocessed image.

        Args:
            img: Image to process
            allowed_chars: String of allowed characters
            use_beam_search: Whether to use beam search decoder (slower but more accurate)

        Returns:
            List of (text, confidence, method) tuples
        """
        try:
            # Convert to RGB for EasyOCR if needed
            if len(img.shape) == 2:  # If grayscale
                img_for_ocr = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
            else:
                img_for_ocr = img

            decoder = 'beamsearch' if use_beam_search else 'greedy'
            decoder_args = {'beamWidth': 5} if use_beam_search else {}

            detection_results = self.reader.readtext(
                img_for_ocr,
                detail=1,  # Return bounding boxes and confidences
                paragraph=False,  # Treat each text box separately
                decoder=decoder,
                allowlist=allowed_chars,
                batch_size=1,  # Process one image at a time
                mag_ratio=1.5,  # Moderate magnification ratio for speed
                canvas_size=1280,  # Smaller canvas size for faster processing
                contrast_ths=0.2,  # Lower contrast threshold for low contrast plates
                adjust_contrast=1.3,  # Moderate contrast adjustment
                **decoder_args
            )

            return detection_results
        except Exception as e:
            logger.debug(f"EasyOCR error with {decoder} decoder: {e}")
            return []

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
            # Apply deskew with a reasonable angle limit for license plates
            plate_img = deskew(plate_img, max_angle_deg=15)

            # Preprocess the plate image for OCR - returns a dictionary of processed images
            processed_images = self._preprocess_plate(plate_img)
            if not processed_images:
                return None, 0.0

            all_results = []
            allowed_chars = 'ABCDEFGHIJKLMNOPQRSTUVWXYZÄÖÜ0123456789-'

            # Process each image variant with EasyOCR
            for img_type, img in processed_images.items():
                # Run OCR with greedy decoder (faster)
                detection_results = self._run_ocr_on_image(
                    img, allowed_chars, use_beam_search=False
                )

                # Process results
                for box, text, confidence in detection_results:
                    if confidence >= confidence_threshold:
                        cleaned_text = self._clean_plate_text(text)
                        if cleaned_text:
                            all_results.append((cleaned_text, confidence, f"easyocr-{img_type}"))

            if len(all_results) <= 1:
                # Try with beam search for higher accuracy on challenging plates
                for img_type in ['enhanced', 'binary', 'adaptive']:
                    img = processed_images.get(img_type)
                    if img is not None:
                        beam_results = self._run_ocr_on_image(
                            img, allowed_chars, use_beam_search=True
                        )
                        for box, text, confidence in beam_results:
                            if confidence >= confidence_threshold:
                                cleaned_text = self._clean_plate_text(text)
                                if cleaned_text:
                                    all_results.append((cleaned_text, confidence, f"easyocr-beam-{img_type}"))

            if len(all_results) == 0:
                logger.debug(f"No text with confidence >= {confidence_threshold} detected")
                return None, 0.0

            # Order by confidence
            all_results.sort(key=lambda x: x[1], reverse=True)

            # Find the first valid license plate format
            for text, conf, method in all_results:
                if self._is_valid_plate(text):
                    logger.debug(f"Valid plate detected: {text} (conf: {conf:.2f}, method: {method})")
                    return text, conf

            # If no valid plates, return the highest confidence result
            best_result = all_results[0]
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
            Tuple of (plate_img, plate_number, plate_confidence, ocr_confidence)
            If no plate is detected, returns (None, None, 0.0, 0.0)
            If plate is detected but no characters recognized, returns (plate_img, None, plate_confidence, 0.0)
        """
        try:
            # Detect license plate
            plate_img, plate_bbox, plate_confidence = self.detect_plate(
                vehicle_img, confidence_threshold)

            if plate_img is None:
                return None, None, 0.0, 0.0

            # Recognize characters on the plate
            plate_number, ocr_confidence = self.recognize_characters(
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

            return plate_img, plate_number, plate_confidence, ocr_confidence

        except Exception as e:
            logger.error(f"Error during license plate recognition: {e}")
            return None, None, 0.0, 0.0
