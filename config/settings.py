import os

class Config:
    # OCR配置
    TESSERACT_PATH = "/usr/bin/tesseract"  # Linux默认路径
    
    # 模型参数
    MIN_FACE_CONFIDENCE = 0.6
    EMOTION_THRESHOLD = 0.2
