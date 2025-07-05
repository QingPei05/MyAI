from deepface import DeepFace
from fer import FER
import cv2
import numpy as np
from typing import Dict

def detect_emotions(image: np.ndarray) -> Dict[str, float]:
    """Hybrid emotion detection using FER and DeepFace"""
    # Initialize detectors
    detector = FER(mtcnn=True)
    
    # Detect faces
    faces = detector.detect_emotions(image)
    if not faces:
        return {"No faces detected": 1.0}
    
    # Analyze each face
    results = {}
    for face in faces:
        roi = _extract_face_roi(image, face['box'])
        analysis = _deepface_analyze(roi)
        for emotion, score in analysis.items():
            results[emotion] = results.get(emotion, 0) + score
    
    return _normalize_results(results)

# ... (helper functions would follow)
