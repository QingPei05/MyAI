import cv2
import numpy as np
from typing import Dict, List

class EmotionDetector:
    def __init__(self):
        """修复版情绪检测器"""
        self.detector_backend = "retinaface"  # 恢复使用但指定版本
        self.models = ["VGG-Face"]  # 使用单一轻量模型
        self.min_confidence = 0.7

    def detect(self, image: np.ndarray) -> Dict[str, float]:
        """安全封装的原生DeepFace调用"""
        try:
            # 验证图像格式
            if len(image.shape) != 3 or image.shape[2] != 3:
                image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
                
            # 安全调用
            from deepface import DeepFace
            results = DeepFace.analyze(
                img_path=image,
                actions=['emotion'],
                detector_backend=self.detector_backend,
                models=self.models,
                enforce_detection=False,
                silent=True
            )
            
            if isinstance(results, list):
                return results[0]['emotion']
            return results['emotion']
            
        except Exception as e:
            return {"error": f"情绪分析失败: {str(e)}"}
