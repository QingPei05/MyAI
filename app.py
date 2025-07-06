import cv2
import numpy as np
import streamlit as st
from PIL import Image

# Load pre-trained models
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')
smile_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_smile.xml')

def detect_emotion(img):
    """Detect emotions (happy/neutral/sad) using OpenCV"""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    
    results = []
    for (x,y,w,h) in faces:
        roi_gray = gray[y:y+h, x:x+w]
        roi_color = img[y:y+h, x:x+w]
        
        # Detect smiles
        smiles = smile_cascade.detectMultiScale(roi_gray, scaleFactor=1.8, minNeighbors=20)
        # Detect eyes
        eyes = eye_cascade.detectMultiScale(roi_gray)
        
        # Emotion logic
        emotion = "neutral"  # Default
        if len(smiles) > 0:
            emotion = "happy"
        elif len(eyes) > 0:
            if eyes[0][1] < h/3:  # Eyes positioned high
                emotion = "sad"
        
        results.append({
            "box": [x,y,w,h],
            "emotion": emotion,
            "landmarks": {
                "eyes": [(x+ex, y+ey, ew, eh) for (ex,ey,ew,eh) in eyes],
                "smiles": [(x+sx, y+sy, sw, sh) for (sx,sy,sw,sh) in smiles]
            }
        })
    
    return results

def draw_detections(img, results):
    """Draw detection results on image"""
    for result in results:
        x,y,w,h = result["box"]
        
        # Face box color by emotion
        color = {
            "happy": (0, 255, 0),    # Green
            "neutral": (255, 255, 0), # Yellow
            "sad": (0, 0, 255)       # Red
        }.get(result["emotion"], (255,255,255))
        
        cv2.rectangle(img, (x,y), (x+w,y+h), color, 2)
        
        # Emotion label
        cv2.putText(img, result["emotion"], (x, y-10), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)
        
        # Draw eyes and smile areas
        for (ex,ey,ew,eh) in result["landmarks"]["eyes"]:
            cv2.rectangle(img, (ex,ey), (ex+ew,ey+eh), (255,0,0), 1)
        for (sx,sy,sw,sh) in result["landmarks"]["smiles"]:
            cv2.rectangle(img, (sx,sy), (sx+sw,sy+sh), (0,255,255), 1)
    
    return img

def main():
    st.set_page_config(page_title="Emotion Detection", layout="wide")
    st.title("😊 Emotion Detection (OpenCV)")
    
    uploaded_file = st.file_uploader("Upload Image (JPG/PNG)", type=["jpg", "png"])
    
    if uploaded_file:
        try:
            # Convert image format
            image = Image.open(uploaded_file)
            img = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
            
            # Detect emotions
            results = detect_emotion(img)
            detected_img = draw_detections(img.copy(), results)
            
            # Display results
            col1, col2 = st.columns(2)
            with col1:
                st.image(image, caption="Original Image", use_column_width=True)
            with col2:
                st.image(detected_img, channels="BGR", caption="Analysis Result", use_column_width=True)
            
            # Text results
            for i, result in enumerate(results):
                st.markdown(f"**Face {i+1}**:")
                st.write(f"- Emotion: `{result['emotion']}`")
                st.write(f"- Position: `{result['box']}`")
                
        except Exception as e:
            st.error(f"Processing error: {str(e)}")

if __name__ == "__main__":
    main()
