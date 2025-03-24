"""
License plate detection and recognition using a pre-trained model.
"""
import os
import logging
import cv2
import numpy as np
import torch

logger = logging.getLogger(__name__)

class PlateRecognizer:
    """
    Detects and recognizes license plates using a pre-trained model.
    """
    
    def __init__(self):
        """
        Initialize the license plate detector and recognizer.
        """
        self.model = None
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        
        try:
            self._load_model()
        except Exception as e:
            logger.error(f"Error loading license plate model: {e}")
            logger.warning("Using fallback plate detection")
    
    def _load_model(self):
        """
        Load the pre-trained license plate detection model.
        """
        try:
            # Try to load the model from torch hub - using YOLOv5s as an example
            # In a real implementation, you would use a specialized license plate model
            self.model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)
            
            # Set model parameters
            self.model.to(self.device)
            self.model.conf = 0.25  # Default confidence threshold
            logger.info("Loaded pre-trained model")
            
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            self.model = None
            raise
    
    def recognize(self, vehicle_img, confidence_threshold=0.7):
        """
        Detect and recognize license plate in a vehicle image.
        
        Args:
            vehicle_img: Vehicle image
            confidence_threshold: Minimum confidence for recognition
            
        Returns:
            Tuple of (plate_img, plate_number, confidence)
            If no plate is detected or recognized, returns (None, None, 0.0)
        """
        if vehicle_img is None or vehicle_img.size == 0:
            return None, None, 0.0
            
        # If model failed to load, use fallback detection
        if self.model is None:
            return self._fallback_detection(vehicle_img)
        
        # Update model confidence threshold
        self.model.conf = confidence_threshold
        
        # Run inference
        results = self.model(vehicle_img)
        
        # Process results - filter for potential license plate objects
        # For a generic model, we'll look for rectangles that might be license plates
        # In a real implementation with a specialized model, you would filter for license plate class
        
        license_plate_classes = [0, 2, 7]  # person, car, truck (potential license plate holders)
        
        # Get all detections
        all_detections = results.xyxy[0].cpu().numpy() if len(results.xyxy) > 0 else []
        
        # Filter for potential license plate classes
        plate_candidates = []
        
        for det in all_detections:
            x1, y1, x2, y2, conf, cls = det
            if int(cls) in license_plate_classes and conf >= confidence_threshold:
                # For now, we'll just take the bottom part of vehicles as potential plate locations
                # In a real implementation, you would use a dedicated license plate detector
                h = y2 - y1
                plate_y1 = y1 + 0.7 * h  # Bottom 30% of the vehicle
                plate_candidate = [x1, plate_y1, x2, y2, conf]
                plate_candidates.append(plate_candidate)
        
        if not plate_candidates:
            # No potential license plate detected
            return None, None, 0.0
        
        # Use the highest confidence detection
        best_candidate = max(plate_candidates, key=lambda x: x[4])
        x1, y1, x2, y2, confidence = best_candidate
        
        # Ensure coordinates are within image boundaries
        height, width = vehicle_img.shape[:2]
        x1 = max(0, int(x1))
        y1 = max(0, int(y1))
        x2 = min(width, int(x2))
        y2 = min(height, int(y2))
        
        # Crop license plate image
        plate_img = vehicle_img[y1:y2, x1:x2]
        
        # For now, we don't have character recognition
        # In a real implementation, you would use a specialized model for plate character recognition
        plate_number = None
        
        return plate_img, plate_number, confidence
    
    def _fallback_detection(self, vehicle_img):
        """
        Fallback method for plate detection when the model is not available.
        Uses basic OpenCV methods.
        
        Args:
            vehicle_img: Vehicle image
            
        Returns:
            Tuple of (plate_img, None, confidence)
        """
        # Convert to grayscale
        gray = cv2.cvtColor(vehicle_img, cv2.COLOR_BGR2GRAY)
        
        # Apply bilateral filter to remove noise
        filtered = cv2.bilateralFilter(gray, 11, 17, 17)
        
        # Find edges
        edged = cv2.Canny(filtered, 30, 200)
        
        # Find contours
        contours, _ = cv2.findContours(edged.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        
        # Sort contours by area and keep the largest ones
        contours = sorted(contours, key=cv2.contourArea, reverse=True)[:10]
        
        plate_contour = None
        
        # Loop through contours to find license plate
        for contour in contours:
            # Approximate the contour
            perimeter = cv2.arcLength(contour, True)
            approx = cv2.approxPolyDP(contour, 0.018 * perimeter, True)
            
            # License plates typically have 4 corners
            if len(approx) == 4:
                plate_contour = approx
                break
                
        if plate_contour is None:
            # If no rectangular contour is found, try different approach
            for contour in contours:
                x, y, w, h = cv2.boundingRect(contour)
                aspect_ratio = w / float(h)
                
                # License plates usually have this aspect ratio
                if 1.5 <= aspect_ratio <= 6.0:
                    plate_contour = contour
                    break
        
        # If no license plate contour found
        if plate_contour is None:
            return None, None, 0.0
            
        # Get coordinates
        x, y, w, h = cv2.boundingRect(plate_contour)
        
        # Crop the plate region
        plate_img = vehicle_img[y:y+h, x:x+w]
        
        # Return plate image without text recognition
        return plate_img, None, 0.5  # Default confidence