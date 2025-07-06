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
    """使用精确规则分析单帧图像的情绪"""
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray = cv2.equalizeHist(gray)
    
    faces = face_cascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(30, 30),
        flags=cv2.CASCADE_SCALE_IMAGE
    )
    
    emotions = []
    for (x, y, w, h) in faces:
        roi_gray = gray[y:y+h, x:x+w]
        
        # 检测面部特征
        smiles = smile_cascade.detectMultiScale(
            roi_gray, 
            scaleFactor=1.8, 
            minNeighbors=20,
            minSize=(25, 25))
        
        eyes = eye_cascade.detectMultiScale(
            roi_gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(20, 20))
        
        eye_count = len(eyes)
        smile_count = len(smiles)
        eye_sizes = [e[2] for e in eyes] if eye_count > 0 else [0]
        avg_eye_size = np.mean(eye_sizes) if eye_sizes else 0
        
        # 精确的情绪判断逻辑
        if smile_count > 3:
            emotions.append("快乐")
        elif smile_count > 1:
            if eye_count >= 2 and avg_eye_size > h * 0.2:
                emotions.append("兴奋")
            else:
                emotions.append("满足")
        elif smile_count > 0:
            emotions.append("平静")
        elif eye_count >= 2:
            if any(e[1] > h * 0.6 for e in eyes):  # 眼睛位置低
                emotions.append("悲伤")
            elif any(e[1] < h * 0.3 for e in eyes):  # 眼睛位置高
                if w > h * 0.85:  # 宽脸
                    emotions.append("愤怒")
                else:
                    emotions.append("骄傲")
            else:
                if avg_eye_size > h * 0.22:  # 大眼睛
                    emotions.append("惊讶")
                else:
                    emotions.append("中性")
        else:
            # 根据脸部特征判断
            if w > h * 0.85:  # 宽脸
                emotions.append("愤怒")
            elif h > w * 1.4:  # 长脸
                emotions.append("悲伤")
            else:
                emotions.append("中性")
    
    return emotions

def process_frame(frame):
    """处理单帧图像并返回标记后的图像"""
    emotions = detect_emotion(frame)
    marked_img = frame.copy()
    
    # 精简后的情绪颜色映射
    emotion_colors = {
        "快乐": (0, 255, 0),      # 绿色
        "悲伤": (0, 0, 255),      # 红色
        "愤怒": (0, 0, 139),     # 深红
        "骄傲": (255, 215, 0),   # 金色
        "兴奋": (255, 165, 0),   # 橙色
        "满足": (60, 179, 113),  # 绿色
        "平静": (173, 216, 230), # 浅蓝
        "惊讶": (255, 255, 0),   # 青色
        "中性": (255, 255, 255)  # 白色
    }
    
    for (x, y, w, h), emotion in zip(
        face_cascade.detectMultiScale(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)),
        emotions
    ):
        color = emotion_colors.get(emotion, (255, 255, 255))
        
        # 绘制人脸框
        cv2.rectangle(marked_img, (x, y), (x+w, y+h), color, 2)
        
        # 在脸旁添加情绪标签（带背景）
        label = f"{emotion}"
        font_scale = 0.9 if w > 60 else 0.7
        thickness = 2 if w > 60 else 1
        
        (label_width, label_height), baseline = cv2.getTextSize(
            label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)
        
        # 标签背景
        cv2.rectangle(marked_img,
                     (x, y - label_height - 10),
                     (x + label_width, y),
                     color, cv2.FILLED)
        
        # 标签文字
        cv2.putText(marked_img, label,
                   (x, y - 10),
                   cv2.FONT_HERSHEY_SIMPLEX,
                   font_scale, (0, 0, 0),  # 黑色文字
                   thickness, cv2.LINE_AA)
    
    return marked_img

def process_video(video_path):
    """处理视频文件"""
    cap = cv2.VideoCapture(video_path)
    stframe = st.empty()
    stop_button = st.button("停止处理")
    
    while cap.isOpened() and not stop_button:
        ret, frame = cap.read()
        if not ret:
            break
            
        marked_frame = process_frame(frame)
        stframe.image(marked_img, channels="BGR")
        
    cap.release()
    if stop_button:
        st.warning("视频处理已中断")

def process_uploaded_file(uploaded_file):
    """自动处理上传的图片或视频"""
    file_type = uploaded_file.type.split('/')[0]
    
    if file_type == "image":
        image = Image.open(uploaded_file)
        img = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
        emotions = detect_emotion(img)
        
        emotion_count = {}
        for e in emotions:
            emotion_count[e] = emotion_count.get(e, 0) + 1
        
        col1, col2 = st.columns([1, 2])
        
        with col1:
            st.subheader("情绪统计")
            if emotion_count:
                # 中文情绪排序
                emotion_order = ["快乐", "兴奋", "满足", "平静", "骄傲", 
                                "惊讶", "中性", "悲伤", "愤怒"]
                sorted_emotions = sorted(emotion_count.items(),
                                       key=lambda x: emotion_order.index(x[0]) 
                result_text = "\n".join([f"• {emotion}: {count}人" for emotion, count in sorted_emotions])
                st.success(result_text)
            else:
                st.warning("未检测到人脸")
        
        with col2:
            tab1, tab2 = st.tabs(["原始图片", "分析结果"])
            with tab1:
                st.image(image, use_column_width=True)
            with tab2:
                marked_img = process_frame(img)
                st.image(marked_img, channels="BGR", use_column_width=True)
    
    elif file_type == "video":
        with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as tmpfile:
            tmpfile.write(uploaded_file.read())
            video_path = tmpfile.name
        
        st.info("视频处理中...")
        process_video(video_path)

def main():
    st.set_page_config(
        page_title="高级情绪检测系统",
        page_icon="😊",
        layout="wide"
    )
    
    st.title("😊 高级情绪分析报告")
    st.caption("上传图片或视频进行多情绪检测分析")
    
    uploaded_file = st.file_uploader(
        "选择图片或视频文件（JPG/PNG/MP4）",
        type=["jpg", "png", "jpeg", "mp4"],
        accept_multiple_files=False
    )
    
    if uploaded_file:
        with st.spinner("分析中，请稍候..."):
            process_uploaded_file(uploaded_file)

if __name__ == "__main__":
    main()
