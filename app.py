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
    # 转换为灰度图并增强对比度
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray = cv2.equalizeHist(gray)
    
    # 改进的人脸检测参数
    faces = face_cascade.detectMultiScale(
        gray,
        scaleFactor=1.1,  # 减少缩放因子以提高检测率
        minNeighbors=5,   # 减少邻居数量以检测更多面孔
        minSize=(30, 30), # 设置更小的最小尺寸
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
            minSize=(25, 25)
        )
        eyes = eye_cascade.detectMultiScale(
            roi_gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(20, 20)
        
        # 扩展的情绪判断逻辑
        eye_count = len(eyes)
        smile_count = len(smiles)
        
        # 眼睛位置分析
        eye_positions = [eye[1] for eye in eyes] if eye_count > 0 else []
        avg_eye_position = sum(eye_positions)/len(eye_positions) if eye_positions else 0
        
        # 情绪判断
        if smile_count > 3:
            emotions.append("快乐")
        elif smile_count > 0:
            if eye_count > 0 and avg_eye_position < h * 0.4:
                emotions.append("快乐")
            else:
                emotions.append("惊讶")
        elif eye_count > 1:
            if avg_eye_position > h * 0.6:
                emotions.append("悲伤")
            elif avg_eye_position < h * 0.3:
                emotions.append("愤怒")
            else:
                emotions.append("恐惧")
        else:
            emotions.append("厌恶")
    
    return emotions

def process_frame(frame):
    """处理单帧图像并返回标记后的图像"""
    emotions = detect_emotion(frame)
    marked_img = frame.copy()
    
    for (x, y, w, h), emotion in zip(
        face_cascade.detectMultiScale(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)),
        emotions
    ):
        color = {
            "快乐": (0, 255, 0),      # 绿色
            "悲伤": (0, 0, 255),      # 红色
            "愤怒": (0, 0, 139),     # 深红色
            "恐惧": (255, 0, 0),     # 蓝色
            "厌恶": (139, 0, 139),   # 紫色
            "惊讶": (255, 255, 0),   # 青色
        }.get(emotion, (255, 255, 255))
        cv2.rectangle(marked_img, (x, y), (x+w, y+h), color, 2)
        cv2.putText(marked_img, emotion, (x, y-10),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)
    return marked_img

def process_video(video_path):
    """处理视频文件"""
    cap = cv2.VideoCapture(video_path)
    stframe = st.empty()
    stop_button = st.button("停止处理")
    
    while cap.isOpened() and not stop_button:
        ret, frame = cap.read()
        if not ret:
            break
            
        marked_frame = process_frame(frame)
        stframe.image(marked_frame, channels="BGR")
        
    cap.release()
    if stop_button:
        st.warning("视频处理已中断")

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
                result_text = "，".join([f"{emotion}: {count}人" for emotion, count in emotion_count.items()])
                st.success(result_text)
            else:
                st.warning("未检测到人脸")
        
        with col2:
            # 并排显示原图和分析结果
            tab1, tab2 = st.tabs(["原始图片", "分析结果"])
            with tab1:
                st.image(image, use_container_width=True)
            with tab2:
                marked_img = process_frame(img)
                st.image(marked_img, channels="BGR", use_container_width=True)
    
    elif file_type == "video":
        # 处理视频
        with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as tmpfile:
            tmpfile.write(uploaded_file.read())
            video_path = tmpfile.name
        
        st.info("视频处理中...")
        process_video(video_path)

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
