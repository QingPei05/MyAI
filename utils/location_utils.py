import pytesseract
import easyocr
import cv2
import re
import numpy as np
from geopy.geocoders import Nominatim
from google.cloud import vision
from typing import List, Optional
from config.settings import Config

class EnhancedLocationDetector:
    def __init__(self):
        self.ocr_reader = easyocr.Reader(['en', 'zh'])
        self.geolocator = Nominatim(user_agent="high-accuracy-geo-detector")
        self.vision_client = None
        
        if Config.GOOGLE_CREDENTIALS:
            self.vision_client = vision.ImageAnnotatorClient()

    def detect(self, image: np.ndarray, use_google: bool = True) -> str:
        """Multi-modal location detection pipeline"""
        # Step 1: Try GPS EXIF data first
        gps_location = self._extract_gps(image)
        if gps_location:
            return gps_location
        
        # Step 2: Hybrid OCR (Tesseract + EasyOCR)
        ocr_text = self._hybrid_ocr(image)
        locations = self._extract_location_candidates(ocr_text)
        
        # Step 3: Google Vision Landmark Detection
        if use_google and self.vision_client:
            landmarks = self._google_landmark_detection(image)
            locations.extend(landmarks)
        
        # Step 4: Validate and geocode
        return self._resolve_location(locations[:3])  # Top 3 candidates

    def _hybrid_ocr(self, image: np.ndarray) -> str:
        """Combine Tesseract and EasyOCR for maximum text coverage"""
        # Preprocess image
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        denoised = cv2.fastNlMeansDenoising(gray, h=30)
        
        # Tesseract with custom config
        custom_config = r'--oem 3 --psm 6'
        text = pytesseract.image_to_string(denoised, config=custom_config)
        
        # EasyOCR for complex cases
        easy_results = self.ocr_reader.readtext(denoised, detail=0)
        text += " ".join(easy_results)
        
        return text

    def _google_landmark_detection(self, image: np.ndarray) -> List[str]:
        """Use Google Vision API for landmark recognition"""
        _, encoded_img = cv2.imencode('.jpg', image)
        content = encoded_img.tobytes()
        image = vision.Image(content=content)
        
        response = self.vision_client.landmark_detection(image=image)
        return [landmark.description for landmark in response.landmark_annotations]

    def _resolve_location(self, candidates: List[str]) -> str:
        """Validate and format the best location"""
        for loc in candidates:
            try:
                location = self.geolocator.geocode(
                    loc, 
                    exactly_one=True,
                    timeout=10,
                    language='en'
                )
                if location:
                    address = location.raw['address']
                    return f"{loc}, {address.get('city', '')}, {address.get('country', '')}"
            except Exception:
                continue
        return "Location not recognized"
