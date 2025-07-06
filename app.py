import cv2
import numpy as np
import streamlit as st
from PIL import Image
import tempfile
from moviepy.editor import VideoFileClip

# 初始化检测器
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')
smile_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_smile.xml')

def detect_emotion(frame):
    """分析单帧图像的情绪"""
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    
    results = []
    for (x, y, w, h) in faces:
        roi_gray = gray[y:y+h, x:x+w]
        
        # 检测微笑和眼睛
        smiles = smile_cascade.detectMultiScale(roi_gray, scaleFactor=1.8, minNeighbors=20)
        eyes = eye_cascade.detectMultiScale(roi_gray)
        
        # 情绪判断逻辑
        emotion = "neutral"
        if len(smiles) > 0:
            emotion = "happy"
        elif len(eyes) > 0:
            if eyes[0][1] < h/3:  # 眼睛位置偏高
                emotion = "sad"
        
        # 绘制检测框
        color = {"happy": (0, 255, 0), "sad": (0, 0, 255)}.get(emotion, (255, 255, 0))
        cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
        cv2.putText(frame, emotion, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)
    
    return frame

def process_video(uploaded_file):
    """处理上传的视频文件"""
    # 保存临时文件
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp:
        tmp.write(uploaded_file.read())
        input_path = tmp.name
    
    # 使用moviepy处理视频
    clip = VideoFileClip(input_path)
    processed_clip = clip.fl_image(detect_emotion)
    
    # 保存结果
    output_path = "output.mp4"
    processed_clip.write_videofile(
        output_path,
        codec="libx264",
        audio=False,
        threads=4,  # 多线程加速
        preset="fast"  # 加速编码
    )
    
    return output_path

def main():
    st.set_page_config(page_title="AI情绪检测", layout="wide")
    st.title("🎭 实时情绪分析系统")
    
    # 模式选择
    analysis_mode = st.radio(
        "选择输入类型",
        ["图片检测", "视频检测"],
        horizontal=True
    )
    
    if analysis_mode == "图片检测":
        uploaded_file = st.file_uploader("上传图片", type=["jpg", "png", "jpeg"])
        if uploaded_file:
            try:
                image = Image.open(uploaded_file)
                img = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
                
                # 处理并显示结果
                result_img = detect_emotion(img.copy())
                col1, col2 = st.columns(2)
                with col1:
                    st.image(image, caption="原始图片", use_container_width=True)
                with col2:
                    st.image(result_img, channels="BGR", caption="分析结果", use_container_width=True)
                    
            except Exception as e:
                st.error(f"图片处理失败: {str(e)}")
    
    else:  # 视频检测模式
        uploaded_file = st.file_uploader("上传视频", type=["mp4", "avi", "mov"])
        if uploaded_file:
            st.video(uploaded_file)
            
            if st.button("开始分析"):
                with st.spinner("视频处理中..."):
                    try:
                        output_path = process_video(uploaded_file)
                        st.success("分析完成！")
                        st.video(output_path)
                        
                        # 提供下载链接
                        with open(output_path, "rb") as f:
                            st.download_button(
                                label="下载结果视频",
                                data=f,
                                file_name="emotion_output.mp4",
                                mime="video/mp4"
                            )
                    except Exception as e:
                        st.error(f"视频处理失败: {str(e)}")

if __name__ == "__main__":
    main()
