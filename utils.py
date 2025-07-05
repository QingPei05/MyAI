from models.location_model import LocationModel
from models.emotion_model import EmotionModel

# 初始化模型
location_model = LocationModel(api_key='AIzaSyBdEwSEP38eh55gtGhS5JkTmtR87K4-1ug')
emotion_model = EmotionModel()

def detect_location(image):
    """
    调用地点检测模型进行检测
    """
    return location_model.detect_location(image)

def detect_emotion(image):
    """
    调用情感检测模型进行检测
    """
    return emotion_model.detect_emotion(image)
