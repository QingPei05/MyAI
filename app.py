import cv2
import numpy as np
import streamlit as st
from PIL import Image
import tempfile
import os
from moviepy.editor import VideoFileClip

# 初始化检测器（快速参数）
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
smile_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_smile.xml')

# 会话状态存储
if 'uploaded_files' not in st.session_state:
    st.session_state.uploaded_files = []

def detect_emotion_fast(img):
    """极速情绪检测"""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.2, minNeighbors=5)
    
    emotions = []
    for (x,y,w,h) in faces:
        roi_gray = gray[y:y+h, x:x+w]
        smiles = smile_cascade.detectMultiScale(roi_gray, scaleFactor=1.7, minNeighbors=15)
        
        if len(smiles) > 3:
            emotions.append("兴奋")
        elif len(smiles) > 0:
            emotions.append("开心")
        else:
            emotions.append("难受")
    
    return emotions

def process_media(file):
    """自动处理图片/视频"""
    file_type = file.type.split('/')[0]
    temp_path = None
    
    try:
        if file_type == "image":
            # 处理图片
            img = Image.open(file)
            cv_img = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
            emotions = detect_emotion_fast(cv_img)
            
            # 存储结果
            st.session_state.uploaded_files.append({
                "name": file.name,
                "type": "image",
                "emotions": emotions,
                "data": img,
                "temp_path": None
            })
            
        elif file_type == "video":
            # 保存临时视频文件
            with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp:
                tmp.write(file.read())
                temp_path = tmp.name
            
            # 快速分析前10秒
            clip = VideoFileClip(temp_path).subclip(0, min(10, VideoFileClip(temp_path).duration))
            emotions = []
            for frame in clip.iter_frames(fps=5):  # 降帧分析
                frame = cv2.resize(frame, (640, 360))
                emotions.extend(detect_emotion_fast(frame))
            
            # 统计情绪
            emotion_count = {}
            for e in emotions:
                emotion_count[e] = emotion_count.get(e, 0) + 1
            
            st.session_state.uploaded_files.append({
                "name": file.name,
                "type": "video",
                "emotions": emotion_count,
                "data": temp_path,
                "temp_path": temp_path
            })
    
    except Exception as e:
        st.error(f"处理失败: {str(e)}")
        if temp_path and os.path.exists(temp_path):
            os.unlink(temp_path)

def delete_file(index):
    """删除指定文件"""
    if st.session_state.uploaded_files[index]["temp_path"] and os.path.exists(st.session_state.uploaded_files[index]["temp_path"]):
        os.unlink(st.session_state.uploaded_files[index]["temp_path"])
    st.session_state.uploaded_files.pop(index)
    st.rerun()

def main():
    st.set_page_config(page_title="极速情绪检测", layout="centered")
    st.title("📸⚡ 媒体情绪快检")
    
    # 文件上传区
    uploaded_file = st.file_uploader(
        "上传图片或视频（JPG/PNG/MP4）",
        type=["jpg", "png", "jpeg", "mp4"],
        accept_multiple_files=False,
        key="file_uploader"
    )
    
    if uploaded_file and uploaded_file not in [f["name"] for f in st.session_state.uploaded_files]:
        process_media(uploaded_file)
        st.rerun()
    
    # 结果显示区
    for i, file in enumerate(st.session_state.uploaded_files):
        with st.container(border=True):
            col1, col2 = st.columns([0.9, 0.1])
            with col1:
                st.subheader(f"📌 {file['name']}")
                
                if file["type"] == "image":
                    # 图片显示
                    st.image(file["data"], caption="上传图片")
                    emotions_text = "，".join(file["emotions"]) if file["emotions"] else "未检测到人脸"
                    st.markdown(f"**情绪分析**: {emotions_text}")
                
                else:
                    # 视频显示
                    st.video(file["data"])
                    emotions_text = "，".join([f"{count}人{emotion}" for emotion, count in file["emotions"].items()])
                    st.markdown(f"**情绪统计**: {emotions_text}")
            
            with col2:
                # 删除按钮
                st.button("🗑️", key=f"del_{i}", on_click=delete_file, args=(i,), 
                         help="删除此文件")

if __name__ == "__main__":
    main()
