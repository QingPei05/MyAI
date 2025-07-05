import os

class Config:
    # 通过环境变量读取配置
    GOOGLE_MAPS_API_KEY = os.getenv("GOOGLE_MAPS_API_KEY", "")
    USE_GPU = os.getenv("USE_GPU", "false").lower() == "true"
    
    # 模型参数
    MIN_CONFIDENCE = float(os.getenv("MIN_CONFIDENCE", 0.7))
