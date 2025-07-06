import cv2
import numpy as np
import streamlit as st
from PIL import Image

# Initialize detectors with more accurate parameters
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
smile_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_smile.xml')
eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')

def detect_emotion(frame):
    """Analyze single frame for emotions with improved detection"""
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # Improved face detection parameters
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
        
        # Improved feature detection
        smiles = smile_cascade.detectMultiScale(
            roi_gray,
            scaleFactor=1.8,
            minNeighbors=20,
            minSize=(25, 25)
        )
        
        eyes = eye_cascade.detectMultiScale(
            roi_gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(20, 20)
        )
        
        # Enhanced emotion logic
        if len(smiles) > 2:  # Reduced threshold for better detection
            if w > 100 and h > 100:  # Only count if face is reasonably large
                emotions.append("excited")
        elif len(smiles) > 0:
            emotions.append("happy")
        elif len(eyes) == 2:  # Exactly two eyes often indicates neutral
            eye_centers = [y + ey + eh/2 for (ex, ey, ew, eh) in eyes]
            avg_eye_height = sum(eye_centers) / len(eye_centers)
            if avg_eye_height / h > 0.4:  # Adjusted eye position threshold
                emotions.append("sad")
            else:
                emotions.append("neutral")
        else:
            emotions.append("neutral")
    
    return emotions, faces

def process_uploaded_image(uploaded_file):
    """Process uploaded image only"""
    image = Image.open(uploaded_file)
    img = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
    emotions, faces = detect_emotion(img)
    
    # Emotion statistics
    emotion_count = {}
    for e in emotions:
        emotion_count[e] = emotion_count.get(e, 0) + 1
    
    # Layout with normal-sized text
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.subheader("情绪统计")
        if emotion_count:
            result_text = "，".join([f"{count}人{emotion}" for emotion, count in emotion_count.items()])
            st.success(f"**检测结果**: {result_text}")
        else:
            st.warning("未检测到人脸")
    
    with col2:
        tab1, tab2 = st.tabs(["原始图片", "分析结果"])
        with tab1:
            st.image(image, use_column_width=True)
        with tab2:
            marked_img = img.copy()
            for (x, y, w, h), emotion in zip(faces, emotions):
                color = {
                    "happy": (0, 255, 0),      # Green
                    "excited": (0, 255, 255),  # Yellow
                    "sad": (255, 0, 0),        # Red
                    "neutral": (255, 255, 0)   # Cyan
                }.get(emotion, (255, 255, 255))
                
                # Draw rectangle
                cv2.rectangle(marked_img, (x, y), (x+w, y+h), color, 2)
                
                # Draw emotion text with normal size
                cv2.putText(marked_img, emotion, (x, y-5),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
            
            st.image(marked_img, channels="BGR", use_column_width=True)

def main():
    st.set_page_config(page_title="情绪检测系统", layout="centered")
    st.title("📊 情绪分析报告")
    
    uploaded_file = st.file_uploader(
        "上传图片（JPG/PNG）", 
        type=["jpg", "png", "jpeg"],
        key="file_uploader"
    )
    
    if uploaded_file:
        process_uploaded_image(uploaded_file)

if __name__ == "__main__":
    main()
