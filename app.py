import cv2
import numpy as np
import streamlit as st
from PIL import Image
import tempfile

# 初始化检测器
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')
smile_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_smile.xml')

def detect_emotion(frame):
    """单帧情绪检测函数（与之前相同）"""
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    results = []
    for (x,y,w,h) in faces:
        roi_gray = gray[y:y+h, x:x+w]
        smiles = smile_cascade.detectMultiScale(roi_gray, scaleFactor=1.8, minNeighbors=20)
        eyes = eye_cascade.detectMultiScale(roi_gray)
        emotion = "neutral"
        if len(smiles) > 0:
            emotion = "happy"
        elif len(eyes) > 0 and eyes[0][1] < h/3:
            emotion = "sad"
        results.append({"box": [x,y,w,h], "emotion": emotion})
    return results

def process_video(video_path):
    """处理视频文件的核心函数"""
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    st.write(f"视频参数: {int(fps)} FPS, 总帧数: {int(cap.get(cv2.CAP_PROP_FRAME_COUNT))}")
    
    # 创建视频输出器
    output_path = "output.mp4"
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, 
                         (int(cap.get(3)), int(cap.get(4))))
    
    # 进度条
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    frame_count = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
            
        # 处理每一帧
        results = detect_emotion(frame)
        for result in results:
            x,y,w,h = result["box"]
            color = {"happy": (0,255,0), "sad": (0,0,255)}.get(result["emotion"], (255,255,0))
            cv2.rectangle(frame, (x,y), (x+w,y+h), color, 2)
            cv2.putText(frame, result["emotion"], (x,y-10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)
        
        out.write(frame)
        frame_count += 1
        progress_bar.progress(frame_count / int(cap.get(cv2.CAP_PROP_FRAME_COUNT)))
        status_text.text(f"已处理 {frame_count} 帧...")
    
    cap.release()
    out.release()
    return output_path

def main():
    st.set_page_config(page_title="视频情绪检测", layout="wide")
    st.title("🎥 视频情绪分析系统")
    
    # 模式选择
    analysis_mode = st.radio(
        "选择输入类型",
        ["图片检测", "视频检测"],
        horizontal=True
    )
    
    if analysis_mode == "图片检测":
        uploaded_file = st.file_uploader("上传图片", type=["jpg", "png"])
        if uploaded_file:
            image = Image.open(uploaded_file)
            img = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
            results = detect_emotion(img)
            detected_img = img.copy()
            for result in results:
                x,y,w,h = result["box"]
                color = {"happy": (0,255,0), "sad": (0,0,255)}.get(result["emotion"], (255,255,0))
                cv2.rectangle(detected_img, (x,y), (x+w,y+h), color, 2)
            st.image(detected_img, channels="BGR", caption="检测结果")
    
    else:  # 视频检测模式
        uploaded_video = st.file_uploader("上传视频", type=["mp4", "avi"])
        if uploaded_video:
            # 保存临时视频文件
            with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp:
                tmp.write(uploaded_video.read())
                video_path = tmp.name
            
            st.video(uploaded_video)
            if st.button("开始分析视频"):
                output_path = process_video(video_path)
                st.success("分析完成！")
                st.video(output_path, format="video/mp4")

if __name__ == "__main__":
    main()
