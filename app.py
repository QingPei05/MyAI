import cv2
import numpy as np
import streamlit as st
from PIL import Image

# Load pre-trained models
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')
smile_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_smile.xml')

def detect_emotion(img):
    """Detect emotions using OpenCV (happy, neutral, sad, angry)"""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    
    emotions = []
    for (x,y,w,h) in faces:
        roi_gray = gray[y:y+h, x:x+w]
        
        # Detect smiles
        smiles = smile_cascade.detectMultiScale(roi_gray, scaleFactor=1.8, minNeighbors=20)
        # Detect eyes
        eyes = eye_cascade.detectMultiScale(roi_gray)
        
        # Emotion detection logic
        emotion = "neutral"  # default
        
        # Anger detection
        if len(eyes) >= 2:
            eye_centers = [y + ey + eh/2 for (ex,ey,ew,eh) in eyes[:2]]
            avg_eye_height = np.mean(eye_centers)
            eye_sizes = [eh for (ex,ey,ew,eh) in eyes[:2]]
            avg_eye_size = np.mean(eye_sizes)
            
            if avg_eye_size > h/5 and avg_eye_height < h/2.5:
                emotion = "angry"
            elif avg_eye_height < h/3:
                emotion = "sad"
        
        # Happiness detection (priority)
        if len(smiles) > 0:
            emotion = "happy"
        
        emotions.append(emotion)
    
    return emotions, faces

def draw_detections(img, emotions, faces):
    """Draw detection boxes with English labels"""
    output_img = img.copy()
    
    # Color mapping
    color_map = {
        "happy": (0, 255, 0),     # green
        "neutral": (255, 255, 0), # yellow
        "sad": (0, 0, 255),       # red
        "angry": (0, 165, 255)    # orange
    }
    
    for i, ((x,y,w,h), emotion) in enumerate(zip(faces, emotions)):
        color = color_map.get(emotion, (255, 255, 255))
        
        # Draw face rectangle
        cv2.rectangle(output_img, (x,y), (x+w,y+h), color, 3)
        
        # Add emotion label
        cv2.putText(output_img, 
                   emotion.upper(), 
                   (x+5, y-10), 
                   cv2.FONT_HERSHEY_SIMPLEX, 
                   0.8, 
                   color, 
                   2)
    
    return output_img

def main():
    st.set_page_config(page_title="Emotion Detection System", layout="wide")
    st.title("😊 Emotion Detection")
    
    uploaded_file = st.file_uploader("Upload Image (JPG/PNG)", type=["jpg", "png"])
    
    if uploaded_file:
        try:
            # Convert image format
            image = Image.open(uploaded_file)
            img = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
            
            # Detect emotions
            emotions, faces = detect_emotion(img)
            detected_img = draw_detections(img.copy(), emotions, faces)
            
            # Two-column layout
            col1, col2 = st.columns([1, 2])
            
            with col1:
                st.subheader("Detection Results")
                if emotions:
                    emotion_count = {
                        "Happy": emotions.count("happy"),
                        "Neutral": emotions.count("neutral"),
                        "Sad": emotions.count("sad"),
                        "Angry": emotions.count("angry")
                    }
                    
                    result = []
                    for emo, cnt in emotion_count.items():
                        if cnt > 0:
                            result.append(f"{cnt} {emo}")
                    st.success(", ".join(result))
                    
                    st.markdown("---")
                    st.markdown("**Detection Logic**:")
                    st.write("""
                    - 😊 Happy: Detected smile
                    - 😠 Angry: Wide open eyes in upper face
                    - 😐 Neutral: Default neutral expression
                    - 😢 Sad: Eyes positioned higher
                    """)
                else:
                    st.warning("No faces detected")
            
            with col2:
                tab1, tab2 = st.tabs(["Original Image", "Analysis Result"])
                with tab1:
                    st.image(image, use_container_width=True)
                with tab2:
                    st.image(detected_img, channels="BGR", use_container_width=True,
                           caption=f"Detected {len(faces)} faces")
                
        except Exception as e:
            st.error(f"Processing error: {str(e)}")

if __name__ == "__main__":
    main()
