import sys
import subprocess
import streamlit as st
from PIL import Image
import numpy as np
import tempfile
import os
from utils.location_utils import EnhancedLocationDetector
from utils.emotion_utils import MultiModelEmotionAnalyzer

# ======================
# 依赖安装保障模块 (修正版)
# ======================
def safe_import(package_name, pip_name=None):
    """安全导入依赖，失败时显示友好错误"""
    try:
        return __import__(package_name)
    except ImportError:
        package_to_install = pip_name or package_name
        st.error(f"缺少必要依赖: {package_to_install}")
        st.info(f"请手动执行: pip install {package_to_install}")
        st.stop()

# 提前导入必要依赖
cv2 = safe_import("cv2", "opencv-python-headless==4.9.0.80")
pd = safe_import("pandas")

# 初始化检测器（放在全局避免重复加载）
@st.cache_resource
def load_detectors():
    return {
        "location": EnhancedLocationDetector(),
        "emotion": MultiModelEmotionAnalyzer()
    }

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
    
    uploaded_file = st.file_uploader(
        "上传图片或视频 (支持 JPG/PNG/MP4)", 
        type=["jpg", "jpeg", "png", "mp4"]
    )

    if uploaded_file:
        detectors = load_detectors()
        if uploaded_file.type.startswith('image'):
            process_image(uploaded_file, detectors)
        elif uploaded_file.type.startswith('video'):
            process_video(uploaded_file, detectors)

# ======================
# 处理函数
# ======================
def process_image(file, detectors):
    try:
        image = Image.open(file).convert('RGB')
        img_array = np.array(image)
        
        col1, col2 = st.columns(2)
        with col1:
            st.image(image, use_column_width=True)
        
        with st.spinner('分析中...'):
            # 实际调用地点检测
            location = detectors["location"].detect(img_array)
            
            # 实际调用情绪检测
            emotions = detectors["emotion"].analyze(img_array)
        
        with col2:
            st.subheader("结果")
            st.markdown(f"**📍 地点**: {location}")
            st.markdown("**😊 情绪**:")
            for e, s in emotions.items():
                st.progress(s, text=f"{e}: {s:.2f}")
                
    except Exception as e:
        st.error(f"图片处理错误: {str(e)}")

def process_video(file, detectors):
    try:
        st.warning("视频处理可能需要较长时间...")
        with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as tmp:
            tmp.write(file.read())
            tmp_path = tmp.name
        
        # 使用OpenCV读取视频
        cap = cv2.VideoCapture(tmp_path)
        frame_count = 0
        results = []
        
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
                
            frame_count += 1
            if frame_count % 5 != 0:  # 每5帧处理一次
                continue
                
            # 转换颜色空间
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # 处理帧
            try:
                location = detectors["location"].detect(frame_rgb)
                emotions = detectors["emotion"].analyze(frame_rgb)
                results.append({
                    "frame": frame_count,
                    "location": location,
                    "emotions": emotions
                })
            except Exception as e:
                st.warning(f"帧 {frame_count} 处理失败: {str(e)}")
            
            # 更新进度
            progress = min(frame_count / 100, 1.0)  # 假设最多处理100帧
            progress_bar.progress(progress)
            status_text.text(f"已处理 {frame_count} 帧...")
        
        cap.release()
        os.unlink(tmp_path)
        
        st.success("分析完成！")
        st.dataframe(pd.DataFrame(results))
        
    except Exception as e:
        st.error(f"视频处理错误: {str(e)}")
    finally:
        if 'tmp_path' in locals() and os.path.exists(tmp_path):
            os.unlink(tmp_path)

if __name__ == "__main__":
    main()
