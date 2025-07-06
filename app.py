import cv2
import numpy as np
import streamlit as st
from PIL import Image
import tempfile
import os
from moviepy.editor import VideoFileClip
import pandas as pd
import matplotlib.pyplot as plt

# 初始化检测器
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
smile_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_smile.xml')
eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')

# 中英文情绪标签
EMOTION_LABELS = {
    "happy": {"en": "happy", "cn": "高兴"},
    "excited": {"en": "excited", "cn": "兴奋"},
    "sad": {"en": "sad", "cn": "悲伤"},
    "neutral": {"en": "neutral", "cn": "平静"}
}

# 情绪对应颜色
EMOTION_COLORS = {
    "happy": (0, 255, 0),      # 绿色
    "excited": (0, 255, 255),  # 黄色
    "sad": (0, 0, 255),        # 红色
    "neutral": (255, 255, 0)   # 青色
}

def detect_emotion(frame):
    """分析单帧图像的情绪"""
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    
    emotions = []
    for (x, y, w, h) in faces:
        roi_gray = gray[y:y+h, x:x+w]
        
        # 检测面部特征
        smiles = smile_cascade.detectMultiScale(roi_gray, scaleFactor=1.8, minNeighbors=20)
        eyes = eye_cascade.detectMultiScale(roi_gray)
        
        # 情绪判断逻辑
        if len(smiles) > 3:
            emotions.append("excited")
        elif len(smiles) > 0:
            emotions.append("happy")
        elif len(eyes) > 0 and eyes[0][1] / h < 0.3:
            emotions.append("sad")
        else:
            emotions.append("neutral")
    
    return emotions

def mark_emotion_on_image(img, emotions):
    """在图片上标记情绪"""
    marked_img = img.copy()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    
    for (x, y, w, h), emotion in zip(faces, emotions):
        color = EMOTION_COLORS.get(emotion, (255, 255, 255))
        cv2.rectangle(marked_img, (x, y), (x+w, y+h), color, 2)
        # 显示中英文标签
        label = f"{EMOTION_LABELS[emotion]['en']}/{EMOTION_LABELS[emotion]['cn']}"
        cv2.putText(marked_img, label, (x, y-10), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
    
    return marked_img

def process_image(uploaded_file):
    """处理上传的图片"""
    image = Image.open(uploaded_file)
    img = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
    emotions = detect_emotion(img)
    
    # 统计情绪
    emotion_count = {e: emotions.count(e) for e in set(emotions)}
    
    # 新布局：左侧统计，右侧图片
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.subheader("📊 情绪统计")
        if emotion_count:
            # 创建饼图
            fig, ax = plt.subplots()
            ax.pie(
                emotion_count.values(),
                labels=[EMOTION_LABELS[e]["cn"] for e in emotion_count.keys()],
                colors=[np.array(EMOTION_COLORS[e])/255 for e in emotion_count.keys()],
                autopct='%1.1f%%'
            )
            st.pyplot(fig)
            
            # 显示统计结果
            result_text = "，".join(
                [f"{count}人{EMOTION_LABELS[emotion]['cn']}" 
                 for emotion, count in emotion_count.items()]
            )
            st.success(f"**检测结果**: {result_text}")
        else:
            st.warning("未检测到人脸")
    
    with col2:
        # 并排显示原图和分析结果
        tab1, tab2 = st.tabs(["原始图片", "分析结果"])
        with tab1:
            st.image(image, use_container_width=True)
        with tab2:
            if emotions:
                marked_img = mark_emotion_on_image(img, emotions)
                st.image(marked_img, channels="BGR", use_container_width=True)
            else:
                st.image(image, use_container_width=True)

def process_video(uploaded_file):
    """处理上传的视频文件"""
    # 创建临时文件
    with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as tmp_file:
        tmp_file.write(uploaded_file.read())
        video_path = tmp_file.name
    
    # 读取视频
    clip = VideoFileClip(video_path)
    duration = clip.duration
    st.info(f"🎥 视频信息: 长度 {duration:.2f}秒, {clip.fps:.2f} FPS")
    
    # 设置采样帧数
    sample_freq = min(2, clip.fps)  # 每秒最多采样2帧
    total_frames = int(duration * sample_freq)
    
    # 进度条
    progress_bar = st.progress(0)
    status_text = st.empty()
    status_text.text("准备开始视频分析...")
    
    # 分析视频帧
    emotions_over_time = []
    sample_frames = []
    
    for i, frame in enumerate(clip.iter_frames(fps=sample_freq)):
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        emotions = detect_emotion(frame)
        emotions_over_time.extend(emotions)
        
        # 每5帧保存一个样本用于展示
        if i % 5 == 0 and emotions:
            marked_frame = mark_emotion_on_image(frame, emotions)
            sample_frames.append(marked_frame)
        
        # 更新进度
        progress = (i + 1) / total_frames
        progress_bar.progress(progress)
        status_text.text(f"分析进度: {int(progress*100)}% 完成 ({i+1}/{total_frames}帧)")
    
    # 显示统计结果
    st.subheader("📈 视频情绪分析报告")
    
    if emotions_over_time:
        # 情绪频率统计
        emotion_count = {e: emotions_over_time.count(e) for e in set(emotions_over_time)}
        
        # 转换为DataFrame便于显示
        df = pd.DataFrame.from_dict(
            {EMOTION_LABELS[e]["cn"]: count for e, count in emotion_count.items()},
            orient='index',
            columns=['出现次数']
        )
        
        # 两列布局
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("### 情绪出现频率")
            st.dataframe(df.style.background_gradient(cmap='Blues'))
            
        with col2:
            st.write("### 情绪分布比例")
            fig, ax = plt.subplots()
            ax.pie(
                df['出现次数'],
                labels=df.index,
                colors=[np.array(EMOTION_COLORS[e])/255 for e in emotion_count.keys()],
                autopct='%1.1f%%'
            )
            st.pyplot(fig)
        
        # 显示样本帧
        st.write("### 视频分析样例")
        cols = st.columns(min(3, len(sample_frames)))
        for idx, frame in enumerate(sample_frames[:3]):
            cols[idx].image(frame, channels="BGR", use_container_width=True)
        
        # 情绪变化趋势图
        st.write("### 情绪变化趋势")
        timeline = pd.DataFrame({
            "时间点": [i/sample_freq for i in range(len(emotions_over_time))],
            "情绪": [EMOTION_LABELS[e]["cn"] for e in emotions_over_time]
        })
        st.line_chart(timeline.groupby(["时间点", "情绪"]).size().unstack().fillna(0))
    else:
        st.warning("⚠️ 视频中未检测到人脸")
    
    # 清理临时文件
    clip.close()
    os.unlink(video_path)

def process_uploaded_file(uploaded_file):
    """自动处理上传的图片或视频"""
    try:
        file_type = uploaded_file.type.split('/')[0]
        
        if file_type == "image":
            process_image(uploaded_file)
        elif file_type == "video":
            process_video(uploaded_file)
        else:
            st.error("不支持的文件类型")
    except Exception as e:
        st.error(f"处理文件时出错: {str(e)}")

def main():
    st.set_page_config(
        page_title="智能情绪分析系统",
        layout="wide",
        page_icon="😊"
    )
    
    st.title("😊 智能情绪分析系统")
    st.markdown("""
        <style>
        .stProgress > div > div > div > div {
            background-color: #1E90FF;
        }
        </style>
    """, unsafe_allow_html=True)
    
    uploaded_file = st.file_uploader(
        "上传图片或视频（JPG/PNG/MP4）", 
        type=["jpg", "png", "jpeg", "mp4"],
        help="支持单人或多人的图片/视频分析"
    )
    
    if uploaded_file:
        st.sidebar.info("文件信息", icon="ℹ️")
        st.sidebar.write(f"文件名: {uploaded_file.name}")
        st.sidebar.write(f"文件类型: {uploaded_file.type}")
        
        with st.spinner("分析中，请稍候..."):
            process_uploaded_file(uploaded_file)

if __name__ == "__main__":
    main()
