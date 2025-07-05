import cv2
import pytesseract
import easyocr
import re
import numpy as np
from geopy.geocoders import Nominatim
from typing import List, Optional, Tuple

class LocationDetector:
    def __init__(self):
        """初始化多模态地点检测器"""
        self.ocr_reader = easyocr.Reader(['en', 'zh'])
        self.geolocator = Nominatim(user_agent="geo-detection-v2")
        self.min_confidence = 0.7

    def detect(self, image: np.ndarray) -> str:
        """
        主检测方法
        :param image: RGB格式的numpy数组
        :return: 地点字符串
        """
        try:
            # 预处理图像
            processed_img = self._preprocess_image(image)
            
            # 多阶段检测
            location = self._detect_by_text(processed_img) or \
                      self._detect_by_landmark(image)
            
            return location or "地点无法识别"

        except Exception as e:
            return f"检测错误: {str(e)}"

    def _preprocess_image(self, image: np.ndarray) -> np.ndarray:
        """图像增强处理"""
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        denoised = cv2.fastNlMeansDenoising(gray, h=30)
        return cv2.threshold(denoised, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]

    def _detect_by_text(self, image: np.ndarray) -> Optional[str]:
        """通过OCR文本识别地点"""
        # 多引擎OCR
        custom_config = r'--oem 3 --psm 6 -l chi_sim+eng'
        text = pytesseract.image_to_string(image, config=custom_config)
        easy_results = self.ocr_reader.readtext(image, detail=0)
        combined_text = text + " " + " ".join(easy_results)
        
        # 提取地点候选
        candidates = self._extract_location_candidates(combined_text)
        return self._resolve_location(candidates[0]) if candidates else None

    def _extract_location_candidates(self, text: str) -> List[str]:
        """从文本提取可能的地点"""
        patterns = [
            r'[\u4e00-\u9fa5]{2,8}(?:市|区|县|镇|村|街道)',  # 中文地名
            r'[A-Z][a-zA-Z\s]{3,}(?:City|Town|Village)',    # 英文地名
            r'[A-Z]{2,}\s*\d{2,5}'                          # 道路编号
        ]
        candidates = []
        for pattern in patterns:
            candidates.extend(re.findall(pattern, text))
        return list(set(candidates))

    def _detect_by_landmark(self, image: np.ndarray) -> Optional[str]:
        """通过视觉地标识别"""
        # 实际项目中应替换为您的视觉地标模型
        # 这里使用简单的颜色特征作为示例
        hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
        if np.mean(hsv[:,:,0]) > 100:  # 示例条件
            return "疑似自然景观区域"
        return None

    def _resolve_location(self, candidate: str) -> str:
        """地理编码解析"""
        try:
            location = self.geolocator.geocode(candidate, exactly_one=True, timeout=10)
            if location:
                return f"{location.address} (经度:{location.longitude:.4f}, 纬度:{location.latitude:.4f})"
            return candidate
        except:
            return candidate
