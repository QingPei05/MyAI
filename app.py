import streamlit as st
from PIL import Image
import numpy as np
from utils.location_detector import detect_location
from utils.emotion_detector import detect_emotions

st.title("🌍 地点与情绪检测系统")
st.write("上传照片或视频，系统将检测拍摄地点和人物情绪")

uploaded_file = st.file_uploader("选择文件", type=["jpg", "jpeg", "png", "mp4"])

if uploaded_file is not None:
    if uploaded_file.type.startswith('image'):
        # 处理图片
        image = Image.open(uploaded_file)
        st.image(image, caption='上传的图片', use_column_width=True)
        
        # 转换为numpy数组供模型处理
        img_array = np.array(image)
        
        with st.spinner('正在分析...'):
            # 检测地点
            location = detect_location(img_array)
            # 检测情绪
            emotions = detect_emotions(img_array)
            
        st.success("分析完成！")
        st.subheader("结果:")
        st.write(f"📍 **地点**: {location}")
        st.write(f"😊 **情绪分析**: {emotions}")
        
    elif uploaded_file.type.startswith('video'):
        # 处理视频（略复杂的实现）
        st.warning("视频处理功能正在开发中...")
