import streamlit as st
import numpy as np
from PIL import Image
from utils.location_utils import LocationDetector
from utils.emotion_utils import EmotionDetector

# 初始化检测器
location_detector = LocationDetector()
emotion_detector = EmotionDetector()

def main():
    st.set_page_config(
        page_title="高精度地点情绪检测系统",
        layout="wide"
    )
    
    st.title("🌍 高精度地点与情绪检测")
    
    uploaded_file = st.file_uploader(
        "上传图片或视频 (支持 JPG/PNG/MP4)", 
        type=["jpg", "jpeg", "png", "mp4"]
    )
    
    if uploaded_file:
        if uploaded_file.type.startswith('image'):
            process_image(uploaded_file)

def process_image(file):
    try:
        image = Image.open(file).convert('RGB')
        img_array = np.array(image)
        
        col1, col2 = st.columns(2)
        with col1:
            st.image(image, caption="上传的图片", use_column_width=True)
        
        with st.spinner('正在分析...'):
            # 实际检测调用
            location = location_detector.detect(img_array)
            emotions = emotion_detector.detect(img_array)
        
        with col2:
            display_results(location, emotions)
            
    except Exception as e:
        st.error(f"处理错误: {str(e)}")

def display_results(location, emotions):
    st.subheader("分析结果")
    
    st.markdown("### 📍 地点识别")
    if isinstance(location, str) and location.startswith("地点检测错误"):
        st.error(location)
    else:
        st.success(location)
    
    st.markdown("### 😊 情绪分析")
    if isinstance(emotions, dict) and "error" in emotions:
        st.error(emotions["error"])
    else:
        for emotion, score in emotions.items():
            st.progress(score, text=f"{emotion}: {score:.2f}")

if __name__ == "__main__":
    main()
