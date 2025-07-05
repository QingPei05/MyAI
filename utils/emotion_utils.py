import cv2
import numpy as np
from deepface import DeepFace
from typing import Dict, List

class EmotionDetector:
    def __init__(self):
        """初始化多模型情绪分析器"""
        self.models = ["VGG-Face", "Facenet", "OpenFace"]
        self.detector_backend = "mtcnn"
        self.min_confidence = 0.65

    def detect(self, image: np.ndarray) -> Dict[str, float]:
        """
        主检测方法
        :param image: RGB格式的numpy数组
        :return: 情绪概率字典
        """
        try:
            faces = self._detect_faces(image)
            if not faces:
                return {"error": "未检测到合格人脸"}
            
            # 多模型集成分析
            ensemble_results = {}
            for face in faces:
                analysis = self._analyze_face(face)
                for emotion, score in analysis.items():
                    ensemble_results[emotion] = ensemble_results.get(emotion, 0) + score
            
            # 归一化结果
            total = sum(ensemble_results.values())
            return {k: round(v/total, 3) for k, v in ensemble_results.items()}
            
        except Exception as e:
            return {"error": f"分析失败: {str(e)}"}

    def _detect_faces(self, image: np.ndarray) -> List[np.ndarray]:
        """人脸检测与对齐"""
        # 使用MTCNN获取更精确的人脸区域
        detections = DeepFace.extract_faces(
            image,
            detector_backend=self.detector_backend,
            enforce_detection=False
        )
        return [face["face"] for face in detections if face["confidence"] > self.min_confidence]

    def _analyze_face(self, face: np.ndarray) -> Dict[str, float]:
        """单张人脸的多模型分析"""
        model_results = DeepFace.analyze(
            face,
            actions=['emotion'],
            detector_backend=self.detector_backend,
            models=self.models,
            enforce_detection=False,
            silent=True
        )
        
        # 合并多模型结果
        combined = {}
        for result in model_results:
            for emotion, score in result["emotion"].items():
                combined[emotion] = combined.get(emotion, 0) + score
        
        return combined
