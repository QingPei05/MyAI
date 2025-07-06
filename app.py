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
    """分析单帧图像的情绪（返回详细结果和标记后的图像）"""
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    
    results = []
    marked_img = frame.copy()
    
    for (x, y, w, h) in faces:
        roi_gray = gray[y:y+h, x:x+w]
        
        # 检测面部特征
        smiles = smile_cascade.detectMultiScale(roi_gray, scaleFactor=1.8, minNeighbors=20)
        eyes = eye_cascade.detectMultiScale(roi_gray)
        
        # 高级情绪判断逻辑
        emotion = "neutral"
        if len(smiles) > 0:
            smile_conf = len(smiles) / (w * h) * 1000  # 微笑密度
            if smile_conf > 0.5:
                emotion = "excited"
            else:
                emotion = "happy"
        elif len(eyes) > 0:
            eye_pos = eyes[0][1] / h  # 眼睛相对位置
            if eye_pos < 0.3:
                emotion = "sad"
            elif eye_pos > 0.7:
                emotion = "surprised"
        
        # 绘制检测框和标签
        color = {
            "excited": (0, 255, 255),  # 黄色
            "happy": (0, 255, 0),      # 绿色
            "sad": (255, 0, 0),        # 蓝色
            "surprised": (255, 0, 255),# 粉色
            "neutral": (255, 255, 0)   # 青色
        }.get(emotion, (255, 255, 255))
        
        cv2.rectangle(marked_img, (x, y), (x+w, y+h), color, 2)
        cv2.putText(marked_img, f"{emotion}", (x, y-10), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)
        
        results.append({
            "position": [int(x), int(y), int(w), int(h)],
            "emotion": emotion,
            "confidence": {
                "smile_density": len(smiles) / (w * h) if len(smiles) > 0 else 0,
                "eye_position": eyes[0][1] / h if len(eyes) > 0 else 0.5
            }
        })
    
    return marked_img, results

def process_uploaded_file(uploaded_file):
    """自动处理上传的图片或视频"""
    file_type = uploaded_file.type.split('/')[0]
    
    if file_type == "image":
        # 处理图片
        image = Image.open(uploaded_file)
        img = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
        marked_img, results = detect_emotion(img)
        
        # 显示结果
        col1, col2 = st.columns(2)
        with col1:
            st.image(image, caption="原始图片", use_container_width=True)
        with col2:
            st.image(marked_img, channels="BGR", caption="分析结果", use_container_width=True)
        
        # 显示详细情绪数据
        st.subheader("情绪分析报告")
        for i, result in enumerate(results):
            st.markdown(f"""
            **人脸 {i+1}**  
            - 情绪: `{result['emotion']}`  
            - 位置: `{result['position']}`  
            - 微笑强度: `{result['confidence']['smile_density']:.2f}`  
            - 眼睛位置: `{result['confidence']['eye_position']:.2f}`
            """)
    
    elif file_type == "video":
        # 处理视频
        st.warning("视频处理可能需要较长时间，请耐心等待...")
        
        # 保存临时文件
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp:
            tmp.write(uploaded_file.read())
            input_path = tmp.name
        
        # 处理视频并显示进度
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        def process_frame(frame):
            marked_frame, _ = detect_emotion(frame)
            progress = min((frame_count / total_frames), 1.0)
            progress_bar.progress(progress)
            status_text.text(f"已处理 {frame_count}/{total_frames} 帧...")
            return marked_frame
        
        clip = VideoFileClip(input_path)
        total_frames = int(clip.fps * clip.duration)
        frame_count = 0
        
        processed_clip = clip.fl_image(lambda f: process_frame(f))
        output_path = "output.mp4"
        processed_clip.write_videofile(output_path, codec="libx264", audio=False)
        
        # 显示结果
        st.success("处理完成！")
        st.video(output_path)
        
        # 提供下载
        with open(output_path, "rb") as f:
            st.download_button(
                label="下载结果视频",
                data=f,
                file_name="emotion_analysis.mp4"
            )

def main():
    st.set_page_config(page_title="智能情绪检测", layout="wide")
    st.title("🎭 AI情绪分析系统")
    
    uploaded_file = st.file_uploader(
        "上传图片或视频（支持JPG/PNG/MP4）", 
        type=["jpg", "png", "jpeg", "mp4"],
        accept_multiple_files=False
    )
    
    if uploaded_file:
        process_uploaded_file(uploaded_file)

if __name__ == "__main__":
    main()
