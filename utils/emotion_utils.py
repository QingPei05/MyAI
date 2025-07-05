import cv2
import numpy as np
from deepface import DeepFace

class EmotionDetector:
    def __init__(self):
        self.models = ["VGG-Face", "Facenet", "OpenFace"]
        
    def detect(self, image):
        """实际情绪检测逻辑"""
        try:
            # 1. 人脸检测
            faces = self._detect_faces(image)
            if not faces:
                return {"error": "未检测到人脸"}
            
            # 2. 多模型集成分析
            results = {}
            for face in faces:
                analysis = DeepFace.analyze(
                    face, 
                    actions=['emotion'],
                    detector_backend='mtcnn',
                    models=self.models,
                    enforce_detection=False
                )
                for model_result in analysis:
                    for emotion, score in model_result['emotion'].items():
                        results[emotion] = results.get(emotion, 0) + score
            
            # 3. 归一化结果
            total = sum(results.values())
            return {k: round(v/total, 3) for k, v in results.items()}
            
        except Exception as e:
            return {"error": str(e)}

    def _detect_faces(self, image):
        """人脸区域提取"""
        face_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        )
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)
        return [image[y:y+h, x:x+w] for (x,y,w,h) in faces]
