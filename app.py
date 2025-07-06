import cv2
import numpy as np
import streamlit as st
from PIL import Image
import pandas as pd
import matplotlib.pyplot as plt

# Load pre-trained models
@st.cache_resource
def load_models():
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')
    smile_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_smile.xml')
    return face_cascade, eye_cascade, smile_cascade

face_cascade, eye_cascade, smile_cascade = load_models()

def detect_emotion(img):
    """Enhanced emotion detection with more robust logic"""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    
    emotions = []
    confidence_scores = []
    for (x,y,w,h) in faces:
        roi_gray = gray[y:y+h, x:x+w]
        
        # Detect facial features
        smiles = smile_cascade.detectMultiScale(
            roi_gray, 
            scaleFactor=1.8, 
            minNeighbors=20,
            minSize=(25, 25)
        )
        eyes = eye_cascade.detectMultiScale(
            roi_gray,
            minSize=(30, 30)
        
        # Initialize emotion with neutral and medium confidence
        emotion = "neutral"
        confidence = 0.5
        
        # Eye-based emotion detection
        if len(eyes) >= 2:
            eye_centers = [y + ey + eh/2 for (ex,ey,ew,eh) in eyes[:2]]
            avg_eye_height = np.mean(eye_centers)
            eye_sizes = [eh for (ex,ey,ew,eh) in eyes[:2]]
            avg_eye_size = np.mean(eye_sizes)
            
            # Anger detection (more precise)
            if avg_eye_size > h/4.5 and avg_eye_height < h/2.7:
                emotion = "angry"
                confidence = 0.7
            # Sad detection (improved threshold)
            elif avg_eye_height < h/3.2:
                emotion = "sad"
                confidence = 0.6
        
        # Happiness detection (priority, with better smile verification)
        if len(smiles) > 0:
            # Only consider smiles in the lower half of the face
            valid_smiles = [s for s in smiles if s[1] > h/2]
            if len(valid_smiles) > 0:
                emotion = "happy"
                confidence = 0.8
        
        emotions.append(emotion)
        confidence_scores.append(confidence)
    
    return emotions, faces, confidence_scores

def draw_detections(img, emotions, faces, confidences):
    """Enhanced visualization with confidence indicators"""
    output_img = img.copy()
    
    # Color and emoji mapping
    color_map = {
        "happy": (0, 255, 0),     # green
        "neutral": (255, 255, 0), # yellow
        "sad": (0, 0, 255),       # red
        "angry": (0, 165, 255)    # orange
    }
    emoji_map = {
        "happy": "😊",
        "neutral": "😐",
        "sad": "😢",
        "angry": "😠"
    }
    
    for i, ((x,y,w,h), emotion, conf) in enumerate(zip(faces, emotions, confidences)):
        color = color_map.get(emotion, (255, 255, 255))
        emoji = emoji_map.get(emotion, "")
        
        # Draw face rectangle
        cv2.rectangle(output_img, (x,y), (x+w,y+h), color, 3)
        
        # Add emotion label with confidence
        label = f"{emoji} {emotion.upper()} ({conf*100:.0f}%)"
        cv2.putText(output_img, 
                   label,
                   (x+5, y-10), 
                   cv2.FONT_HERSHEY_SIMPLEX, 
                   0.8, 
                   color, 
                   2)
    
    return output_img

def show_emotion_stats(emotions):
    """Display emotion statistics with visualization"""
    if not emotions:
        return
    
    st.subheader("📊 Emotion Distribution")
    emotion_count = {
        "Happy": emotions.count("happy"),
        "Neutral": emotions.count("neutral"),
        "Sad": emotions.count("sad"),
        "Angry": emotions.count("angry")
    }
    
    # Create dataframe for visualization
    df = pd.DataFrame.from_dict(emotion_count, orient='index', columns=['Count'])
    df = df[df['Count'] > 0]  # Only show detected emotions
    
    if not df.empty:
        col1, col2 = st.columns([1, 2])
        with col1:
            st.dataframe(df, use_container_width=True)
        
        with col2:
            fig, ax = plt.subplots()
            df.plot(kind='bar', ax=ax, color=['#4CAF50', '#FFC107', '#F44336', '#FF9800'])
            ax.set_title("Emotion Distribution")
            ax.set_ylabel("Number of Faces")
            st.pyplot(fig)
    else:
        st.warning("No emotions detected")

def show_detection_guide():
    """Show detection guide in expandable section"""
    with st.expander("ℹ️ How Emotion Detection Works", expanded=False):
        st.markdown("""
        **Detection Logic Explained:**
        
        - 😊 **Happy**: Detected when smile is present in lower face region
        - 😠 **Angry**: Detected when eyes are wide open and positioned in upper face
        - 😐 **Neutral**: Default state when no strong indicators found
        - 😢 **Sad**: Detected when eyes are positioned higher than normal
        
        **Confidence Scores:**
        - 80-100%: Strong indicators present
        - 60-79%: Moderate confidence
        - Below 60%: Weak indicators
        
        **Tips for Better Results:**
        - Use clear, front-facing images
        - Ensure good lighting
        - Avoid obstructed faces
        """)

def main():
    st.set_page_config(
        page_title="Enhanced Emotion Detection", 
        layout="wide",
        page_icon="😊"
    )
    
    # Custom CSS for better styling
    st.markdown("""
    <style>
    .st-emotion-caption {
        font-size: 16px !important;
        text-align: center !important;
    }
    .stImage > img {
        border-radius: 10px;
    }
    .st-bb {
        background-color: #f0f2f6;
    }
    </style>
    """, unsafe_allow_html=True)
    
    st.title("😊 Enhanced Emotion Detection System")
    st.markdown("Upload an image to detect emotions using computer vision")
    
    # File uploader with more options
    uploaded_file = st.file_uploader(
        "Choose an image...", 
        type=["jpg", "jpeg", "png"],
        help="Select a clear image containing faces"
    )
    
    if uploaded_file:
        try:
            # Convert image format
            image = Image.open(uploaded_file)
            img = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
            
            # Detect emotions with confidence scores
            emotions, faces, confidences = detect_emotion(img)
            detected_img = draw_detections(img.copy(), emotions, faces, confidences)
            
            # Main layout
            col1, col2 = st.columns([1, 2])
            
            with col1:
                st.subheader("🔍 Detection Results")
                
                if emotions:
                    # Display summary
                    result = []
                    for emo, cnt in zip(emotions, confidences):
                        result.append(f"{emo.capitalize()} ({cnt*100:.0f}%)")
                    
                    st.success(f"Detected {len(faces)} face(s): " + ", ".join(result))
                    
                    # Show statistics
                    show_emotion_stats(emotions)
                    
                    # Show detection guide
                    show_detection_guide()
                else:
                    st.warning("No faces detected in the image")
                    st.image(image, use_container_width=True, caption="Uploaded Image")
            
            with col2:
                tab1, tab2 = st.tabs(["📷 Original Image", "🔬 Analysis Result"])
                
                with tab1:
                    st.image(image, 
                            use_container_width=True,
                            caption="Original Image")
                
                with tab2:
                    if faces:
                        st.image(detected_img, 
                               channels="BGR", 
                               use_container_width=True,
                               caption=f"Detected {len(faces)} face(s) with emotions")
                    else:
                        st.image(image,
                               use_container_width=True,
                               caption="No faces detected (original image)")
            
            # Add download button for analyzed image
            if faces:
                st.markdown("---")
                analyzed_img = Image.fromarray(cv2.cvtColor(detected_img, cv2.COLOR_BGR2RGB))
                st.download_button(
                    label="⬇️ Download Analyzed Image",
                    data=cv2.imencode('.jpg', detected_img)[1].tobytes(),
                    file_name="analyzed_emotion.jpg",
                    mime="image/jpeg"
                )
        
        except Exception as e:
            st.error(f"⚠️ Processing error: {str(e)}")
            st.info("Please try another image or check if the file is valid")

if __name__ == "__main__":
    main()
