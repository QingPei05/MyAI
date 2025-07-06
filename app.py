import cv2
import numpy as np
import streamlit as st
from PIL import Image

# Load pre-trained models
@st.cache_resource
def load_models():
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')
    smile_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_smile.xml')
    return face_cascade, eye_cascade, smile_cascade

face_cascade, eye_cascade, smile_cascade = load_models()

def detect_emotion(img):
    """Detect emotions with confidence levels"""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    
    emotions = []
    confidences = []
    for (x,y,w,h) in faces:
        roi_gray = gray[y:y+h, x:x+w]
        
        # Detect facial features
        smiles = smile_cascade.detectMultiScale(roi_gray, scaleFactor=1.8, minNeighbors=20)
        eyes = eye_cascade.detectMultiScale(roi_gray)
        
        # Initialize with neutral and medium confidence
        emotion = "neutral"
        confidence = 0.5  # Default confidence for neutral
        
        # Anger detection
        if len(eyes) >= 2:
            eye_centers = [y + ey + eh/2 for (ex,ey,ew,eh) in eyes[:2]]
            avg_eye_height = np.mean(eye_centers)
            eye_sizes = [eh for (ex,ey,ew,eh) in eyes[:2]]
            avg_eye_size = np.mean(eye_sizes)
            
            if avg_eye_size > h/5 and avg_eye_height < h/2.5:
                emotion = "angry"
                confidence = 0.7  # Higher confidence for clear anger signs
            elif avg_eye_height < h/3:
                emotion = "sad"
                confidence = 0.6  # Moderate confidence for sadness
        
        # Happiness detection (priority)
        if len(smiles) > 0:
            emotion = "happy"
            confidence = 0.8  # High confidence for detected smiles
        
        emotions.append(emotion)
        confidences.append(confidence)
    
    return emotions, faces, confidences

def draw_detections(img, emotions, faces, confidences):
    """Draw detection boxes with labels and confidence"""
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

def show_detection_guide():
    """Show detection guide with confidence explanation"""
    with st.expander("ℹ️ How Emotion Detection Works", expanded=False):
        st.markdown("""
        **Detection Logic Explained:**
        
        - 😊 **Happy** (80% confidence): Detected when smile is present
        - 😠 **Angry** (70% confidence): Eyes wide open and positioned in upper face
        - 😐 **Neutral** (50% confidence): Default state when no strong indicators found
        - 😢 **Sad** (60% confidence): Eyes positioned higher than normal
        
        **Confidence Levels:**
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
        page_title="Emotion Detection System", 
        layout="wide",
        page_icon="😊"
    )
    
    st.title("😊 Emotion Detection with Confidence Levels")
    
    uploaded_file = st.file_uploader("Upload Image (JPG/PNG)", type=["jpg", "png"])
    
    if uploaded_file:
        try:
            # Convert image format
            image = Image.open(uploaded_file)
            img = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
            
            # Detect emotions with confidence
            emotions, faces, confidences = detect_emotion(img)
            detected_img = draw_detections(img.copy(), emotions, faces, confidences)
            
            # Two-column layout
            col1, col2 = st.columns([1, 2])
            
            with col1:
                st.subheader("🔍 Detection Results")
                if emotions:
                    # Display summary with confidence
                    result = []
                    for emo, conf in zip(emotions, confidences):
                        result.append(f"{emo.capitalize()} ({conf*100:.0f}%)")
                    st.success(f"Detected {len(faces)} face(s): " + ", ".join(result))
                    
                    # Show detection guide
                    show_detection_guide()
                else:
                    st.warning("No faces detected")
            
            with col2:
                tab1, tab2 = st.tabs(["Original Image", "Analysis Result"])
                with tab1:
                    st.image(image, use_container_width=True)
                with tab2:
                    st.image(detected_img, channels="BGR", use_container_width=True,
                           caption=f"Detected {len(faces)} faces with confidence levels")
                
        except Exception as e:
            st.error(f"Processing error: {str(e)}")

if __name__ == "__main__":
    main()
