import sys
from pathlib import Path
import streamlit as st
import numpy as np
from PIL import Image
import tempfile
import os

# 解决模块导入问题
sys.path.append(str(Path(__file__).parent))

try:
    from utils.location_utils import LocationDetector
    from utils.emotion_utils import EmotionDetector
except ImportError as e:
    st.error(f"模块导入失败: {str(e)}")
    st.stop()

# 初始化检测器
@st.cache_resource
def load_models():
    return {
        "location": LocationDetector(),
        "emotion": EmotionDetector()
    }

def main():
    st.set_page_config(
        page_title="高精度地点情绪检测系统",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    st.title("🌍 高精度地点与情绪检测")
    st.markdown("""
    <style>
    .stProgress > div > div > div > div {
        background-color: #1E90FF;
    }
    .st-emotion-cache-1kyxreq {
        display: flex;
        justify-content: center;
    }
    </style>
    """, unsafe_allow_html=True)

    # 文件上传
    uploaded_file = st.file_uploader(
        "上传图片或视频 (支持 JPG/PNG/MP4)", 
        type=["jpg", "jpeg", "png", "mp4"],
        help="建议包含清晰文字或地标的媒体文件"
    )

    if uploaded_file:
        if uploaded_file.type.startswith('image'):
            process_image(uploaded_file)
        elif uploaded_file.type.startswith('video'):
            process_video(uploaded_file)

def process_image(file):
    """处理图片文件"""
    try:
        models = load_models()
        image = Image.open(file).convert('RGB')
        img_array = np.array(image)
        
        col1, col2 = st.columns(2)
        with col1:
            st.image(image, caption="上传的图片", use_column_width=True)
        
        with st.spinner('正在分析...'):
            location = models["location"].detect(img_array)
            emotions = models["emotion"].detect(img_array)
        
        with col2:
            display_results(location, emotions)
            
    except Exception as e:
        st.error(f"图片处理错误: {str(e)}")

def process_video(file):
    """处理视频文件"""
    try:
        models = load_models()
        st.warning("视频处理可能需要较长时间，请耐心等待...")
        
        with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as tmp:
            tmp.write(file.read())
            video_path = tmp.name
        
        # 这里添加实际视频处理逻辑
        results = []
        cap = cv2.VideoCapture(video_path)
        frame_count = 0
        
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
                
            if frame_count % 5 == 0:  # 每5帧处理一次
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                location = models["location"].detect(rgb_frame)
                emotions = models["emotion"].detect(rgb_frame)
                results.append({
                    "frame": frame_count,
                    "location": location,
                    "emotions": emotions
                })
                
                # 更新进度
                progress = min(frame_count / 100, 1.0)
                progress_bar.progress(progress)
                status_text.text(f"已处理 {frame_count} 帧...")
            
            frame_count += 1
        
        cap.release()
        os.unlink(video_path)
        
        st.success("分析完成！")
        display_video_results(results)
        
    except Exception as e:
        st.error(f"视频处理错误: {str(e)}")

def display_results(location, emotions):
    """显示图片分析结果"""
    st.subheader("分析结果")
    
    st.markdown("### 📍 地点识别")
    if isinstance(location, str) and location.startswith("检测错误"):
        st.error(location)
    else:
        st.success(location)
    
    st.markdown("### 😊 情绪分析")
    if isinstance(emotions, dict) and "error" in emotions:
        st.error(emotions["error"])
    else:
        for emotion, score in emotions.items():
            st.progress(score, text=f"{emotion}: {score:.2f}")

def display_video_results(results):
    """显示视频分析结果"""
    st.subheader("视频分析报告")
    
    # 显示关键帧摘要
    st.dataframe(
        pd.DataFrame(results),
        use_container_width=True,
        height=300
    )
    
    # 情绪变化趋势图
    st.line_chart(
        pd.DataFrame(results).set_index('frame')['emotions'].apply(pd.Series),
        height=400
    )

if __name__ == "__main__":
    try:
        import pandas as pd
    except ImportError:
        st.warning("正在安装pandas...")
        import subprocess
        subprocess.run([sys.executable, "-m", "pip", "install", "pandas"])
        st.experimental_rerun()
    
    main()
