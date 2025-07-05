import pytesseract
import easyocr
from geopy.geocoders import Nominatim
import cv2
import re
import os
from typing import Optional, List
from config import config

# Initialize OCR readers
reader = easyocr.Reader(['en', 'zh']) if config.USE_EASYOCR else None
geolocator = Nominatim(user_agent="geo-emotion-detector")

def detect_location(image: np.ndarray, use_google_api: bool = True) -> str:
    """
    Multi-stage location detection:
    1. EXIF GPS data
    2. OCR text analysis
    3. Visual landmark recognition
    4. Geocoding
    """
    # Try EXIF first
    gps_info = _extract_exif(image)
    if gps_info:
        return _reverse_geocode(gps_info[0], gps_info[1])
    
    # Multi-engine OCR
    ocr_text = _multimodal_ocr(image)
    locations = _extract_location_keywords(ocr_text)
    
    # Google Vision API
    if use_google_api and config.GOOGLE_CREDENTIALS:
        landmarks = _google_landmark_detection(image)
        if landmarks:
            locations.extend(landmarks)
    
    # Validate and format
    return _resolve_location(locations)

# ... (additional helper functions would follow)
