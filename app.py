import sys
import subprocess
import streamlit as st
from PIL import Image
import numpy as np
import cv2

# ======================
# 依赖自动安装保障模块
# ======================
def install_and_import(package):
    """自动安装缺失的依赖包"""
    try:
        __import__(package.split('==')[0])
    except ImportError:
        st.warning(f"正在安装依赖: {package}...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", package])
        st.success("依赖安装完成！请刷新页面")
        st.experimental_rerun()

# 确保核心依赖存在
REQUIRED_PACKAGES = [
    "opencv-python-headless==4.9.0.80",
    "streamlit==1.32.0",
    "Pillow==10.2.0",
    "numpy==1.26.4"
]

for package in REQUIRED_PACKAGES:
    install_and_import(package)

# ======================
# 应用主界面
# ======================
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

    # ======================
    # 文件上传区域
    # ======================
    uploaded_file = st.file_uploader(
        "上传图片或视频 (支持 JPG/PNG/MP4)", 
        type=["jpg", "jpeg", "png", "mp4"],
        help="建议包含清晰文字或地标的媒体文件"
    )

    if uploaded_file is not None:
        if uploaded_file.type.startswith('image'):
            process_image(uploaded_file)
        elif uploaded_file.type.startswith('video'):
            process_video(uploaded_file)

# ======================
# 图片处理函数
# ======================
def process_image(uploaded_file):
    """处理上传的图片"""
    try:
        image = Image.open(uploaded_file).convert('RGB')
        img_array = np.array(image)
        
        col1, col2 = st.columns(2)
        with col1:
            st.image(image, caption="上传的图片", use_column_width=True)
        
        with st.spinner('正在分析...'):
            # 地点检测
            location = detect_location(img_array)
            
            # 情绪检测
            emotions = detect_emotions(img_array)
        
        with col2:
            display_results(location, emotions)
            
    except Exception as e:
        st.error(f"处理图片时出错: {str(e)}")

# ======================
# 视频处理函数
# ======================
def process_video(uploaded_file):
    """处理上传的视频"""
    try:
        st.warning("视频处理可能需要较长时间，请耐心等待...")
        
        # 创建临时文件
        with st.spinner('准备视频分析...'):
            tfile = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4')
            tfile.write(uploaded_file.read())
            video_path = tfile.name
        
        # 分析视频
        with st.spinner('分析视频帧...'):
            results = analyze_video_frames(video_path)
            os.unlink(video_path)  # 删除临时文件
        
        st.success("分析完成！")
        display_video_results(results)
        
    except Exception as e:
        st.error(f"处理视频时出错: {str(e)}")

# ======================
# 核心检测函数 (需实现)
# ======================
def detect_location(image_array):
    """地点检测逻辑"""
    # 这里应该调用你的地点检测模块
    # 示例实现：
    try:
        # 使用OpenCV进行简单处理
        gray = cv2.cvtColor(image_array, cv2.COLOR_RGB2GRAY)
        # 这里应该是你的实际地点检测代码
        return "示例地点: 北京天安门"
    except:
        return "无法识别地点"

def detect_emotions(image_array):
    """情绪检测逻辑"""
    # 这里应该调用你的情绪检测模块
    # 示例实现：
    try:
        # 这里应该是你的实际情绪检测代码
        return {"happy": 0.7, "neutral": 0.3}
    except:
        return {"error": "情绪检测失败"}

def analyze_video_frames(video_path):
    """分析视频帧"""
    # 这里应该调用你的视频处理模块
    # 示例实现：
    return [
        {"frame": 1, "location": "地点1", "emotions": {"happy": 0.8}},
        {"frame": 2, "location": "地点1", "emotions": {"happy": 0.7}}
    ]

# ======================
# 结果显示函数
# ======================
def display_results(location, emotions):
    """显示分析结果"""
    st.subheader("分析结果")
    
    st.markdown("### 📍 地点识别")
    if location:
        st.success(location)
    else:
        st.warning("无法识别地点")
    
    st.markdown("### 😊 情绪分析")
    if isinstance(emotions, dict):
        for emotion, score in emotions.items():
            st.progress(score, text=f"{emotion}: {score:.2f}")
    else:
        st.warning(emotions)

def display_video_results(results):
    """显示视频分析结果"""
    st.subheader("视频分析报告")
    
    # 显示数据表格
    st.dataframe(
        pd.DataFrame(results),
        use_container_width=True,
        height=300
    )
    
    # 显示情绪变化图表
    emotion_df = pd.DataFrame(results).explode('emotions')
    st.line_chart(
        emotion_df.groupby('frame')['emotions'].value_counts().unstack(),
        height=400
    )

# ======================
# 主程序入口
# ======================
if __name__ == "__main__":
    # 导入额外依赖（仅在需要时）
    try:
        import pandas as pd
        import tempfile
        import os
    except ImportError:
        st.warning("正在安装额外依赖...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "pandas"])
        st.experimental_rerun()
    
    main()
