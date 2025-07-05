import streamlit as st
import cv2
import numpy as np
from PIL import Image
from utils.location_utils import detect_location
from utils.emotion_utils import detect_emotions
from utils.video_processor import process_video
import tempfile
import os

# 页面配置
st.set_page_config(page_title="Geo-Emotion Analyzer", layout="wide")
st.title("🌍 高精度地点与情绪检测系统")
st.markdown("""
<style>
.stProgress > div > div > div > div {
    background-color: #1E90FF;
}
</style>
""", unsafe_allow_html=True)

# 文件上传
uploaded_file = st.file_uploader(
    "上传图片或视频 (支持 JPG/PNG/MP4)", 
    type=["jpg", "jpeg", "png", "mp4"],
    help="建议包含清晰文字或地标的媒体文件"
)

if uploaded_file is not None:
    if uploaded_file.type.startswith('image'):
        # 处理图片
        image = Image.open(uploaded_file).convert('RGB')
        img_array = np.array(image)
        
        col1, col2 = st.columns(2)
        with col1:
            st.image(image, caption="原始图片", use_column_width=True)
        
        with st.spinner('正在分析...'):
            # 并行处理
            with st.expander("高级选项", expanded=False):
                enable_google_api = st.checkbox("启用Google Vision API", True)
                enable_emotion = st.checkbox("启用情绪检测", True)
            
            location = detect_location(img_array, enable_google_api)
            if enable_emotion:
                emotions = detect_emotions(img_array)
        
        # 显示结果
        with col2:
            st.subheader("分析结果")
            st.markdown(f"""
            ### 📍 地点识别
            **{location if location else "未识别到有效地点"}**
            
            ### 😊 情绪分析
            {emotions if enable_emotion else "未启用情绪检测"}
            """)
            
    elif uploaded_file.type.startswith('video'):
        # 处理视频
        st.warning("视频处理可能需要较长时间，请耐心等待...")
        with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as tmp:
            tmp.write(uploaded_file.read())
            video_path = tmp.name
        
        results = process_video(video_path)
        os.unlink(video_path)
        
        st.success("分析完成！")
        st.dataframe(results, height=300)
        st.line_chart(results['emotions'].value_counts())
