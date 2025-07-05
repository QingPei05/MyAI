import streamlit as st
import cv2
import numpy as np
from utils import detect_location, detect_emotion

# Streamlit 应用标题
st.title("AI Location and Emotion Detection")

# 用户上传文件
uploaded_file = st.file_uploader("上传照片或视频...", type=["jpg", "jpeg", "png", "mp4"])

if uploaded_file is not None:
    # 处理图片或视频
    if uploaded_file.type in ["image/jpeg", "image/png"]:
        image = np.array(bytearray(uploaded_file.read()), dtype=np.uint8)
        img = cv2.imdecode(image, cv2.IMREAD_COLOR)
        st.image(img, caption='Uploaded Image', use_column_width=True)

        location = detect_location(img)  # 检测地点
        emotions = detect_emotion(img)    # 检测情感

        # 显示结果
        st.write(f"Location: {location}")
        st.write(f"Emotion: {emotions}")

    elif uploaded_file.type == "video/mp4":
        video_file = uploaded_file.read()
        st.video(video_file)

        # 视频处理逻辑
        video_path = "temp_video.mp4"
        with open(video_path, "wb") as f:
            f.write(video_file)

        cap = cv2.VideoCapture(video_path)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        locations = []
        emotions = []

        for _ in range(min(frame_count, 10)):  # 只处理前10帧
            ret, frame = cap.read()
            if not ret:
                break
            
            location = detect_location(frame)  # 检测地点
            emotion = detect_emotion(frame)    # 检测情感

            locations.append(location)
            emotions.append(emotion)
        
        cap.release()

        # 显示结果
        st.write(f"Locations detected: {locations}")
        st.write(f"Emotions detected: {emotions}")
