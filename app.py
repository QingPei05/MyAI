import cv2
import numpy as np
import streamlit as st
from PIL import Image

# 初始化检测器
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')
smile_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_smile.xml')

def detect_emotion(frame):
    """检测9种基本情绪"""
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(
        gray, 
        scaleFactor=1.1, 
        minNeighbors=5, 
        minSize=(30, 30)
    )
    
    emotions = []
    for (x, y, w, h) in faces:
        roi_gray = gray[y:y+h, x:x+w]
        
        # 检测面部特征
        eyes = eye_cascade.detectMultiScale(roi_gray)
        smiles = smile_cascade.detectMultiScale(roi_gray, scaleFactor=1.8, minNeighbors=20)
        
        # 情绪判断逻辑
        if len(smiles) > 0:
            if w > 100 and h > 100 and len(smiles) > 2:
                emotions.append("快乐")
            else:
                emotions.append("平静")
        elif len(eyes) == 2:
            eye_centers = [y + ey + eh/2 for (ex, ey, ew, eh) in eyes]
            avg_eye_height = sum(eye_centers) / len(eye_centers)
            if avg_eye_height / h > 0.4:
                emotions.append("悲伤")
            else:
                emotions.append("愤怒")
        else:
            emotions.append("平静")
    
    return emotions, faces

def process_image(uploaded_file):
    """处理上传的图片"""
    image = Image.open(uploaded_file)
    img = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
    emotions, faces = detect_emotion(img)
    
    # 情绪统计
    emotion_count = {e: emotions.count(e) for e in set(emotions)}
    
    # 显示结果
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.subheader("情绪统计")
        if emotion_count:
            result_text = "，".join([f"{count}人{emotion}" for emotion, count in emotion_count.items()])
            st.success(f"**检测结果**: {result_text}")
        else:
            st.warning("未检测到人脸")
    
    with col2:
        tab1, tab2 = st.tabs(["原始图片", "分析结果"])
        with tab1:
            st.image(image, use_container_width=True)
        with tab2:
            marked_img = img.copy()
            for (x, y, w, h), emotion in zip(faces, emotions):
                color = {
                    "快乐": (0, 255, 0),      # 绿色
                    "悲伤": (255, 0, 0),      # 蓝色
                    "愤怒": (0, 0, 255),      # 红色
                    "平静": (255, 255, 255)   # 白色
                }.get(emotion, (255, 255, 255))
                
                cv2.rectangle(marked_img, (x, y), (x+w, y+h), color, 2)
                cv2.putText(marked_img, emotion, (x, y-5), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
            
            st.image(marked_img, channels="BGR", use_container_width=True)

def main():
    st.set_page_config(page_title="基本情绪检测系统", layout="centered")
    st.title("📊 基本情绪分析报告")
    
    uploaded_file = st.file_uploader(
        "上传图片（JPG/PNG）", 
        type=["jpg", "png", "jpeg"],
        key="file_uploader"
    )
    
    if uploaded_file:
        process_image(uploaded_file)

if __name__ == "__main__":
    main()
