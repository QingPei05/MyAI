import streamlit as st
import cv2
import numpy as np
from PIL import Image
from utils.location_utils import EnhancedLocationDetector
from utils.emotion_utils import MultiModelEmotionAnalyzer
from utils.video_processor import VideoAnalyzer
from config.settings import Config

# Initialize detectors
location_detector = EnhancedLocationDetector()
emotion_analyzer = MultiModelEmotionAnalyzer()
video_processor = VideoAnalyzer(location_detector, emotion_analyzer)

st.set_page_config(
    page_title="High-Accuracy Geo-Emotion Analysis",
    layout="wide",
    initial_sidebar_state="expanded"
)

def main():
    st.title("🌍 High-Precision Location & Emotion Detection")
    
    with st.sidebar:
        st.header("Settings")
        Config.USE_GOOGLE_VISION = st.checkbox("Use Google Vision API", True)
        Config.MIN_CONFIDENCE = st.slider("Minimum Confidence", 0.1, 1.0, 0.7)
    
    uploaded_file = st.file_uploader(
        "Upload Image/Video", 
        type=["jpg", "jpeg", "png", "mp4"],
        help="For best results, use clear images with visible text/landmarks"
    )
    
    if uploaded_file:
        if uploaded_file.type.startswith('image'):
            process_image(uploaded_file)
        else:
            process_video(uploaded_file)

def process_image(uploaded_file):
    image = Image.open(uploaded_file).convert('RGB')
    img_array = np.array(image)
    
    with st.spinner('Running high-accuracy analysis...'):
        col1, col2 = st.columns(2)
        
        with col1:
            st.image(image, caption="Original", use_column_width=True)
        
        with col2:
            location = location_detector.detect(
                img_array, 
                use_google=Config.USE_GOOGLE_VISION
            )
            emotions = emotion_analyzer.analyze(
                img_array, 
                min_confidence=Config.MIN_CONFIDENCE
            )
            
            st.subheader("Analysis Results")
            st.success(f"📍 **Location**: {location}")
            st.success(f"😊 **Emotions**: {emotions}")

def process_video(uploaded_file):
    with st.spinner('Processing video frames...'):
        results = video_processor.analyze_video(uploaded_file)
        
        st.dataframe(results)
        st.line_chart(results['emotion'].value_counts())

if __name__ == "__main__":
    main()
