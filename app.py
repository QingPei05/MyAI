import cv2
import numpy as np
import streamlit as st
from moviepy.editor import VideoFileClip
import tempfile

# 初始化检测器（使用更快的参数）
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
smile_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_smile.xml')

def detect_emotion_fast(frame):
    """极速情绪检测（仅返回情绪标签）"""
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)  # 更快的参数
    
    emotions = []
    for (x, y, w, h) in faces:
        roi_gray = gray[y:y+h, x:x+w]
        smiles = smile_cascade.detectMultiScale(roi_gray, scaleFactor=1.7, minNeighbors=15)
        
        # 极简情绪判断
        if len(smiles) > 3:  # 检测到多个微笑区域
            emotions.append("兴奋")
        elif len(smiles) > 0:
            emotions.append("开心")
        else:
            emotions.append("难受")
    
    return emotions

def process_video_fast(uploaded_file):
    """10秒内完成的视频处理"""
    # 保存临时文件
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp:
        tmp.write(uploaded_file.read())
        input_path = tmp.name
    
    # 读取视频并降帧率
    clip = VideoFileClip(input_path)
    if clip.duration > 10:  # 如果视频超过10秒，截取前10秒
        clip = clip.subclip(0, 10)
    
    # 进一步降低处理帧率（5FPS）
    processed_frames = []
    emotions_report = {}
    
    for i, frame in enumerate(clip.iter_frames(fps=5)):  # 降帧处理
        frame = cv2.resize(frame, (640, 360))  # 降低分辨率
        emotions = detect_emotion_fast(frame)
        
        # 统计情绪
        for emotion in emotions:
            emotions_report[emotion] = emotions_report.get(emotion, 0) + 1
        
        # 只保留每5帧的1帧用于输出视频（进一步加速）
        if i % 5 == 0:
            marked_frame = frame.copy()
            for (x, y, w, h), emotion in zip(
                face_cascade.detectMultiScale(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)),
                emotions
            ):
                color = {"开心": (0,255,0), "兴奋": (0,255,255), "难受": (0,0,255)}.get(emotion, (255,255,255))
                cv2.rectangle(marked_frame, (x,y), (x+w,y+h), color, 2)
                cv2.putText(marked_frame, emotion, (x,y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
            processed_frames.append(marked_frame)
    
    # 生成简短视频结果（1FPS）
    if processed_frames:
        from moviepy.video.io.ImageSequenceClip import ImageSequenceClip
        output_clip = ImageSequenceClip(processed_frames, fps=1)
        output_path = "output.mp4"
        output_clip.write_videofile(output_path, codec="libx264", audio=False)
    else:
        output_path = None
    
    # 生成情绪报告文本
    report_text = "，".join([f"{count}人{emotion}" for emotion, count in emotions_report.items()])
    
    return output_path, report_text

def main():
    st.set_page_config(page_title="极速情绪检测", layout="centered")
    st.title("⚡ 10秒情绪快检")
    
    uploaded_file = st.file_uploader("上传视频（MP4/AVI，建议10秒内）", type=["mp4", "avi"])
    
    if uploaded_file:
        if st.button("开始极速分析"):
            with st.spinner("10秒快速分析中..."):
                output_path, report_text = process_video_fast(uploaded_file)
                
            st.success("分析完成！")
            st.markdown(f"**检测结果**: {report_text}")
            
            if output_path:
                st.video(output_path)
                with open(output_path, "rb") as f:
                    st.download_button("下载快检视频", f, "emotion_preview.mp4")

if __name__ == "__main__":
    main()
