"""
License plate detection and character recognition using YOLOv8 for detection and fast-plate-ocr for text recognition.
Optimized for speed and accuracy with European license plates.
"""
import logging
import numpy as np
import torch
from fast_plate_ocr import ONNXPlateRecognizer

from safewheels.functions import to_grayscale

logger = logging.getLogger(__name__)


def get_best_plate_text(texts, confidences):
    """
    Select the best license plate text based on char confidence scores.

    Args:
        texts: List of license plate texts
        confidences: List of character confidence scores

    Returns:
        Tuple of (best license plate text, best confidence score)
    """
    best_text = None
    best_confidence = 0.0

    for text, weights in zip(texts, confidences):
        trimmed_text = text.rstrip("_")
        trimmed_weights = weights[:len(trimmed_text)]

        text_confidence = np.mean(trimmed_weights)
        if text_confidence > best_confidence:
            best_text = trimmed_text
            best_confidence = text_confidence

    return best_text, float(best_confidence)


class EUPlateRecognizer:
    """
    Recognizes license plates using fast-plate-ocr.
    Optimized for EU license plates with fast processing.
    """

    def __init__(self, use_gpu=True, device=None):
        """
        Initialize the license plate detector and character recognizer.

        Args:
            use_gpu: Whether to use GPU for detection and recognition
            device: Specific device to use (cuda:0, cuda:1, mps, cpu)
        """
        self.ocr_recognizer: ONNXPlateRecognizer = None
        self.use_gpu = use_gpu
        self.device = device
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

    def _recognize_characters(self, plate_imgs, confidence_threshold=0.3):
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
            # The fast-plate-ocr expects a grayscale image
            grayscaled_imgs = [to_grayscale(img) for img in plate_imgs]

            # Run with confidence information
            results = self.ocr_recognizer.run(grayscaled_imgs, return_confidence=True)
            plate_text, confidence = get_best_plate_text(results[0], results[1])

            logger.debug(f"OCR result: {plate_text} with confidence {confidence:.2f}")

            # Skip if confidence is too low
            if confidence < confidence_threshold:
                logger.debug(f"OCR confidence too low: {confidence:.2f} < {confidence_threshold}")
                return None, 0.0

        except Exception as e:
            logger.error(f"fast-plate-ocr recognition error: {e}")
            return None, 0.0

        return plate_text, confidence

    def recognize(self, plate_imgs, ocr_confidence_threshold=0.3):
        """
        Recognize license plate contents from one or more plate images.

        Args:
            plate_imgs: List of license plate images
            ocr_confidence_threshold: Minimum confidence for character recognition

        Returns:
        """
        try:
            # Recognize characters on the plate
            plate_number, ocr_confidence = self._recognize_characters(plate_imgs, ocr_confidence_threshold)

            if plate_number:
                # Log the successful detection and recognition
                logger.info(f"Recognized license plate: {plate_number}, OCR: {ocr_confidence:.2f})")
            else:
                # If characters weren't recognized
                plate_number = None
                ocr_confidence = 0.0
                logger.debug("License plate detected but characters not recognized")

            return plate_number, ocr_confidence

        except Exception as e:
            logger.error(f"Error during license plate recognition: {e}")
            return None, 0.0
