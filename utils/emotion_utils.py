import cv2
import numpy as np
from typing import Dict, List

class EmotionDetector:
    def __init__(self):
        """兼容版情绪检测器"""
        self.detector_backend = "opencv"  # 改用OpenCV代替RetinaFace
        self.min_confidence = 0.65

    def detect(self, image: np.ndarray) -> Dict[str, float]:
        """
        兼容版检测方法
        :param image: RGB格式的numpy数组
        :return: 情绪概率字典
        """
        try:
            # 使用OpenCV的人脸检测
            faces = self._detect_faces_opencv(image)
            if not faces:
                return {"error": "未检测到合格人脸"}
            
            # 单模型分析（避免复杂依赖）
            analysis = self._analyze_face_simple(faces[0])
            return self._normalize_results(analysis)
            
        except Exception as e:
            return {"error": f"分析失败: {str(e)}"}

    def _detect_faces_opencv(self, image: np.ndarray) -> List[np.ndarray]:
        """使用OpenCV检测人脸"""
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        face_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        )
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)
        return [image[y:y+h, x:x+w] for (x,y,w,h) in faces]

    def _analyze_face_simple(self, face: np.ndarray) -> Dict[str, float]:
        """简化版情绪分析"""
        # 实际项目中应替换为您的轻量级模型
        # 这里返回示例数据
        return {
            "happy": 0.7,
            "neutral": 0.2,
            "surprise": 0.1
        }

    def _normalize_results(self, results: Dict[str, float]) -> Dict[str, float]:
        """归一化结果"""
        total = sum(results.values())
        return {k: round(v/total, 3) for k, v in results.items()}
