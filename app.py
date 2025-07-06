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
    """分析单帧图像的情绪（仅返回情绪标签）"""
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    
    emotions = []
    for (x, y, w, h) in faces:
        roi_gray = gray[y:y+h, x:x+w]
        
        # 检测面部特征
        smiles = smile_cascade.detectMultiScale(roi_gray, scaleFactor=1.8, minNeighbors=20)
        eyes = eye_cascade.detectMultiScale(roi_gray)
        
        # 情绪判断逻辑
        if len(smiles) > 3:
            emotions.append("excited")
        elif len(smiles) > 0:
            emotions.append("happy")
        elif len(eyes) > 0 and eyes[0][1] / h < 0.3:
            emotions.append("sad")
        else:
            emotions.append("neutral")
    
    return emotions

def process_uploaded_file(uploaded_file):
    """自动处理上传的图片或视频"""
    file_type = uploaded_file.type.split('/')[0]
    
    if file_type == "image":
        # 处理图片
        image = Image.open(uploaded_file)
        img = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
        emotions = detect_emotion(img)
        
        # 统计情绪
        emotion_count = {}
        for e in emotions:
            emotion_count[e] = emotion_count.get(e, 0) + 1
        
        # 新布局：左侧统计，右侧图片
        col1, col2 = st.columns([1, 2])
        
        with col1:
            st.subheader("情绪统计")
            if emotion_count:
                result_text = "，".join([f"{count}人{emotion}" for emotion, count in emotion_count.items()])
                st.success(f"**检测结果**: {result_text}")
            else:
                st.warning("未检测到人脸")
        
        with col2:
            # 并排显示原图和分析结果
            tab1, tab2 = st.tabs(["原始图片", "分析结果"])
            with tab1:
                st.image(image, use_container_width=True)
            with tab2:
                marked_img = img.copy()
                for (x, y, w, h), emotion in zip(
                    face_cascade.detectMultiScale(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)),
                    emotions
                ):
                    color = {
                        "happy": (0, 255, 0),      # 绿色
                        "excited": (0, 255, 255),  # 黄色
                        "sad": (0, 0, 255),        # 红色
                        "neutral": (255, 255, 0)   # 青色
                    }.get(emotion, (255, 255, 255))
                    cv2.rectangle(marked_img, (x, y), (x+w, y+h), color, 2)
                    cv2.putText(marked_img, emotion, (x, y-10),  # 添加英文情绪标签
                               cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)
                st.image(marked_img, channels="BGR", use_cintainer_width=True)

def main():
    st.set_page_config(page_title="情绪检测系统", layout="centered")
    st.title("📊 情绪分析报告")
    
    uploaded_file = st.file_uploader(
        "上传图片或视频（JPG/PNG/MP4）", 
        type=["jpg", "png", "jpeg", "mp4"],
        key="file_uploader"
    )
    
    if uploaded_file:
        process_uploaded_file(uploaded_file)

if __name__ == "__main__":
    main()
