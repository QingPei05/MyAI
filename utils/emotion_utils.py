from deepface import DeepFace
from fer import FER
import cv2
import numpy as np
from typing import Dict, List
from config.settings import Config

class MultiModelEmotionAnalyzer:
    def __init__(self):
        self.face_detector = FER(mtcnn=True)
        self.emotion_models = [
            'DeepFace',
            'Facenet',
            'OpenFace'
        ]
        
    def analyze(self, image: np.ndarray, min_confidence: float = 0.7) -> Dict[str, float]:
        """Ensemble emotion analysis with multiple models"""
        # Face detection
        faces = self._detect_faces(image)
        if not faces:
            return {"No faces detected": 1.0}
        
        # Multi-model analysis
        results = {}
        for face in faces:
            roi = self._extract_face_roi(image, face['box'])
            emotions = self._ensemble_analysis(roi)
            
            for emotion, score in emotions.items():
                if score >= min_confidence:
                    results[emotion] = results.get(emotion, 0) + score
        
        return self._normalize_results(results)

    def _ensemble_analysis(self, face_roi: np.ndarray) -> Dict[str, float]:
        """Combine predictions from multiple models"""
        ensemble_results = {}
        
        # DeepFace analysis
        try:
            df_analysis = DeepFace.analyze(
                face_roi, 
                actions=['emotion'],
                detector_backend='mtcnn',
                enforce_detection=False,
                silent=True
            )
            if isinstance(df_analysis, list):
                df_analysis = df_analysis[0]
            for emotion, score in df_analysis['emotion'].items():
                ensemble_results[emotion] = ensemble_results.get(emotion, 0) + score
        except:
            pass
            
        # FER analysis
        fer_analysis = self.face_detector.detect_emotions(face_roi)
        if fer_analysis:
            for emotion, score in fer_analysis[0]['emotions'].items():
                ensemble_results[emotion] = ensemble_results.get(emotion, 0) + score
        
        return ensemble_results

    def _normalize_results(self, results: Dict[str, float]) -> Dict[str, float]:
        """Convert scores to percentages"""
        total = sum(results.values())
        return {k: round(v/total, 3) for k, v in results.items()}
