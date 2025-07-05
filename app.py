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

    elif uploaded_file.type == "video/mp4":
        video_file = uploaded_file.read()
        # 视频处理逻辑
        st.video(video_file)

        # 这里需要添加视频帧提取和处理逻辑
        # 例如：使用 cv2.VideoCapture 来读取每一帧

    # 显示结果
    st.write(f"Location: {location}")
    st.write(f"Emotion: {emotions}")
