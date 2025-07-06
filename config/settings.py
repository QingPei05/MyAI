class Config:
    # 直接填写您的API密钥（仅用于开发测试）
    GOOGLE_MAPS_API_KEY = "AIzaSyCOIA8yE_qT-VbYqC1rK_jw0Dh_7N3UmcA"
    
    USE_GPU = os.getenv("USE_GPU", "false").lower() == "true"
    MIN_CONFIDENCE = float(os.getenv("MIN_CONFIDENCE", 0.7))
