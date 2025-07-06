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
        if len(smiles) > 3:  # 多个微笑区域
            emotions.append("兴奋")
        elif len(smiles) > 0:
            emotions.append("开心")
        elif len(eyes) > 0 and eyes[0][1] / h < 0.3:  # 眼睛位置偏高
            emotions.append("难受")
        else:
            emotions.append("中性")
    
    return emotions

def process_uploaded_file(uploaded_file):
    """自动处理上传的图片或视频"""
    file_type = uploaded_file.type.split('/')[0]
    
    if file_type == "image":
        # 处理图片
        image = Image.open(uploaded_file)
        img = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
        emotions = detect_emotion(img)
        
        # 显示结果
        col1, col2 = st.columns(2)
        with col1:
            st.image(image, caption="原始图片", use_container_width=True)
        
        # 统计情绪
        emotion_count = {}
        for e in emotions:
            emotion_count[e] = emotion_count.get(e, 0) + 1
        
        # 简化输出格式
        with col2:
            st.image(img, channels="BGR", caption="分析结果", use_container_width=True)
            st.subheader("情绪统计")
            if emotion_count:
                result_text = "，".join([f"{count}人{emotion}" for emotion, count in emotion_count.items()])
                st.success(f"**检测结果**: {result_text}")
            else:
                st.warning("未检测到人脸")

    elif file_type == "video":
        # 处理视频
        st.warning("视频处理中...（自动截取前10秒）")
        
        # 保存临时文件
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp:
            tmp.write(uploaded_file.read())
            input_path = tmp.name
        
        # 只处理前10秒
        clip = VideoFileClip(input_path).subclip(0, min(10, VideoFileClip(input_path).duration))
        total_emotions = []
        
        # 降帧处理（5FPS）
        for frame in clip.iter_frames(fps=5):
            frame = cv2.resize(frame, (640, 360))  # 降低分辨率加速处理
            total_emotions.extend(detect_emotion(frame))
        
        # 统计全局情绪
        emotion_count = {}
        for e in total_emotions:
            emotion_count[e] = emotion_count.get(e, 0) + 1
        
        # 显示结果
        st.success("分析完成！")
        if emotion_count:
            result_text = "，".join([f"{count}人{emotion}" for emotion, count in emotion_count.items()])
            st.markdown(f"**最终统计**: {result_text}")
            
            # 显示示例帧
            st.video(input_path)
        else:
            st.warning("视频中未检测到人脸")

def main():
    st.set_page_config(page_title="极简情绪检测", layout="centered")
    st.title("😊 情绪快检系统")
    
    uploaded_file = st.file_uploader(
        "上传图片或视频（JPG/PNG/MP4）", 
        type=["jpg", "png", "jpeg", "mp4"],
        key="file_uploader"
    )
    
    if uploaded_file:
        process_uploaded_file(uploaded_file)

if __name__ == "__main__":
    main()
