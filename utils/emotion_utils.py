from deepface import DeepFace
from fer import FER
import cv2
import numpy as np
from typing import Dict, Union

def detect_emotions(image: np.ndarray) -> Dict[str, float]:
    """
    混合情绪检测策略：
    1. 先用MTCNN检测人脸（FER）
    2. 用DeepFace进行精细情绪分析
    """
    # 人脸检测
    detector = FER(mtcnn=True)
    faces = detector.detect_emotions(image)
    
    if not faces:
        return {"未检测到人脸": 1.0}
    
    # 多脸情绪分析
    results = {}
    for face in faces:
        roi = image[
            max(0, face['box'][1]):min(image.shape[0], face['box'][1]+face['box'][3]),
            max(0, face['box'][0]):min(image.shape[1], face['box'][0]+face['box'][2])
        ]
        
        # DeepFace精细分析
        analysis = DeepFace.analyze(
            roi, 
            actions=['emotion'], 
            detector_backend='mtcnn',
            enforce_detection=False
        )
        
        for emotion, score in analysis[0]['emotion'].items():
            results[emotion] = results.get(emotion, 0) + score
    
    # 归一化
    total = sum(results.values())
    return {k: round(v/total, 3) for k, v in results.items()}
