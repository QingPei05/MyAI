import streamlit as st
import cv2
import numpy as np
from PIL import Image
from utils.location_utils import detect_location
from utils.emotion_utils import detect_emotions
from utils.video_processor import process_video
import tempfile
import os
from config import config

# App configuration
st.set_page_config(
    page_title="Geo-Emotion Analyzer",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Main UI
st.title("🌍 Location and Emotion Detection")
st.markdown("""
Analyze images/videos to detect:
- 📍 Location from text/landmarks
- 😊 Facial emotions
""")

# File upload
uploaded_file = st.file_uploader(
    "Upload media (JPG/PNG/MP4)",
    type=["jpg", "jpeg", "png", "mp4"]
)

if uploaded_file:
    if uploaded_file.type.startswith('image'):
        process_image(uploaded_file)
    elif uploaded_file.type.startswith('video'):
        process_video_file(uploaded_file)

def process_image(uploaded_file):
    """Handle image processing"""
    image = Image.open(uploaded_file).convert('RGB')
    img_array = np.array(image)
    
    col1, col2 = st.columns(2)
    with col1:
        st.image(image, use_column_width=True)
    
    with st.spinner('Analyzing...'):
        location = detect_location(img_array, config.USE_GOOGLE_API)
        emotions = detect_emotions(img_array)
    
    with col2:
        show_results(location, emotions)

def process_video_file(uploaded_file):
    """Handle video processing"""
    st.warning("Video processing may take several minutes...")
    with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as tmp:
        tmp.write(uploaded_file.read())
        video_path = tmp.name
    
    results = process_video(video_path, config)
    os.unlink(video_path)
    
    st.success("Analysis complete!")
    st.dataframe(results)
    st.line_chart(results['emotions'].value_counts())

def show_results(location, emotions):
    """Display analysis results"""
    st.subheader("Results")
    st.markdown(f"""
    ### 📍 Location
    **{location if location else "No location detected"}**
    
    ### 😊 Emotions
    {emotions if emotions else "No faces detected"}
    """)
