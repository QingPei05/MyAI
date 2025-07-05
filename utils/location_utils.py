import pytesseract
import easyocr
from geopy.geocoders import Nominatim
from google.cloud import vision
import cv2
import re
import os
from typing import Optional

# 初始化EasyOCR和Google Vision
reader = easyocr.Reader(['en', 'zh'])
geolocator = Nominatim(user_agent="geo-emotion-detector")

def detect_location(image: np.ndarray, use_google_api: bool = True) -> str:
    """
    多模态地点识别流程：
    1. 优先尝试EXIF GPS数据
    2. 使用OCR提取文字
    3. 调用Google Vision地标识别
    4. 地理编码解析
    """
    # 尝试读取EXIF
    if hasattr(image, '_getexif'):
        gps_info = _extract_exif_gps(image)
        if gps_info:
            return _reverse_geocode(gps_info[0], gps_info[1])
    
    # 多引擎OCR
    ocr_text = _multimodal_ocr(image)
    potential_locations = _extract_location_keywords(ocr_text)
    
    # Google Vision API（需设置GOOGLE_APPLICATION_CREDENTIALS环境变量）
    if use_google_api and os.getenv('GOOGLE_APPLICATION_CREDENTIALS'):
        landmarks = _google_landmark_detection(image)
        if landmarks:
            potential_locations.extend(landmarks)
    
    # 地理编码解析
    for loc in potential_locations[:3]:  # 最多尝试前3个候选
        try:
            location = geolocator.geocode(loc, addressdetails=True, timeout=10)
            if location:
                addr = location.raw['address']
                return f"{loc}, {addr.get('city', '')}, {addr.get('country', '')}"
        except:
            continue
    
    return potential_locations[0] if potential_locations else "无法识别地点"

def _multimodal_ocr(image: np.ndarray) -> str:
    """组合使用Tesseract和EasyOCR提高文字识别率"""
    # Tesseract（适合清晰文字）
    text = pytesseract.image_to_string(image)
    
    # EasyOCR（适合复杂背景）
    if not text.strip():
        results = reader.readtext(image)
        text = " ".join([res[1] for res in results])
    
    return text

def _google_landmark_detection(image: np.ndarray) -> list:
    """使用Google Vision地标识别"""
    client = vision.ImageAnnotatorClient()
    _, encoded_img = cv2.imencode('.jpg', image)
    content = encoded_img.tobytes()
    
    image = vision.Image(content=content)
    response = client.landmark_detection(image=image)
    
    return [landmark.description for landmark in response.landmark_annotations]
