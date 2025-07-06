import cv2
import numpy as np
import streamlit as st
from PIL import Image
import tempfile
from moviepy.editor import VideoFileClip

# 初始化检测器
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
smile_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_smile.xml')
eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')

def detect_emotion(frame):
    """分析单帧图像的情绪（使用精确规则判断）"""
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray = cv2.equalizeHist(gray)
    
    faces = face_cascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(30, 30),
        flags=cv2.CASCADE_SCALE_IMAGE
    )
    
    emotions = []
    for (x, y, w, h) in faces:
        roi_gray = gray[y:y+h, x:x+w]
        
        # 检测面部特征
        smiles = smile_cascade.detectMultiScale(
            roi_gray, 
            scaleFactor=1.8, 
            minNeighbors=20,
            minSize=(25, 25))
        
        eyes = eye_cascade.detectMultiScale(
            roi_gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(20, 20))
        
        eye_count = len(eyes)
        smile_count = len(smiles)
        eye_positions = [eye[1] for eye in eyes] if eye_count > 0 else []
        avg_eye_position = sum(eye_positions)/len(eye_positions) if eye_positions else 0
        
        # 精确的情绪判断逻辑
        if smile_count > 3:
            if eye_count > 1 and avg_eye_position < h * 0.3:
                emotions.append("爱")
            else:
                emotions.append("快乐")
        elif smile_count > 1:
            if eye_count > 1 and eyes[0][3] > h * 0.2:  # 大眼睛
                emotions.append("兴奋")
            else:
                emotions.append("满足")
        elif smile_count > 0:
            emotions.append("平静")
        elif eye_count > 1:
            if avg_eye_position > h * 0.6:
                if eyes[0][3] < h * 0.15:  # 小眼睛
                    emotions.append("悲伤")
                else:
                    emotions.append("羞愧")
            elif avg_eye_position < h * 0.3:
                if w > h * 0.8:  # 宽脸
                    emotions.append("愤怒")
                else:
                    emotions.append("嫉妒")
            else:
                if eyes[0][3] > h * 0.2:  # 大眼睛
                    emotions.append("恐惧")
                else:
                    emotions.append("焦虑")
        elif eye_count == 1:
            emotions.append("尴尬")
        else:
            # 无特征时的保守判断
            if w > h * 0.8:  # 宽脸
                emotions.append("骄傲")
            elif h > w * 1.3:  # 长脸
                emotions.append("内疚")
            else:
                emotions.append("中性")
    
    return emotions

def process_frame(frame):
    """处理单帧图像并返回标记后的图像"""
    emotions = detect_emotion(frame)
    marked_img = frame.copy()
    
    # 定义情绪颜色映射
    emotion_colors = {
        "快乐": (0, 255, 0),      # 绿色
        "悲伤": (0, 0, 255),      # 红色
        "愤怒": (0, 0, 139),     # 深红色
        "恐惧": (255, 0, 0),     # 蓝色
        "厌恶": (139, 0, 139),   # 紫色
        "惊讶": (255, 255, 0),   # 青色
        "爱": (255, 105, 180),  # 粉色
        "内疚": (165, 42, 42),   # 棕色
        "羞愧": (218, 165, 32),  # 金色
        "尴尬": (255, 192, 203), # 粉红色
        "骄傲": (255, 215, 0),   # 金色
        "羡慕": (50, 205, 50),   # 浅绿色
        "嫉妒": (34, 139, 34),   # 森林绿
        "焦虑": (75, 0, 130),    # 靛蓝色
        "兴奋": (255, 165, 0),   # 橙色
        "平静": (173, 216, 230), # 浅蓝色
        "怀旧": (186, 85, 211),  # 中紫色
        "同情": (0, 191, 255),   # 深天蓝色
        "满足": (60, 179, 113),  # 中海绿色
        "中性": (255, 255, 255)  # 白色
    }
    
    for (x, y, w, h), emotion in zip(
        face_cascade.detectMultiScale(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)),
        emotions
    ):
        color = emotion_colors.get(emotion, (255, 255, 255))
        
        # 绘制人脸矩形框
        cv2.rectangle(marked_img, (x, y), (x+w, y+h), color, 2)
        
        # 在脸旁绘制情绪标签
        label_size, _ = cv2.getTextSize(emotion, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
        cv2.rectangle(marked_img, 
                     (x, y - label_size[1] - 10), 
                     (x + label_size[0], y), 
                     color, cv2.FILLED)
        cv2.putText(marked_img, emotion, (x, y-10),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    
    return marked_img

# 其余函数保持不变（process_video, process_uploaded_file, main）
# ...

if __name__ == "__main__":
    main()
