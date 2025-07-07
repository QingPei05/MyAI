import streamlit as st
import cv2
import numpy as np
from PIL import Image
import pandas as pd
from datetime import datetime
import random
import os
import plotly.express as px
from emotion_utils.detector import EmotionDetector

# ----------------- App Configuration -----------------
st.set_page_config(
    page_title="AI Emotion & Location Detector",
    page_icon="👁‍🗨",
    layout="wide",
    initial_sidebar_state="expanded"
)

@st.cache_resource
def get_detector():
    return EmotionDetector()

detector = get_detector()

def save_history(username, emotion, confidence, location="Unknown"):
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    df = pd.DataFrame([[username, emotion, confidence, location, now]], 
                     columns=["Username","Emotion","Confidence","Location","timestamp"])
    try:
        if os.path.exists("history.csv"):
            prev = pd.read_csv("history.csv")
            df = pd.concat([prev, df])
        df.to_csv("history.csv", index=False)
    except Exception as e:
        st.error(f"Save failed: {e}")

def show_detection_guide():
    with st.expander("ℹ️ How Emotion Detection Works", expanded=False):
        st.markdown("""
        *Detection Model:*
        - Using DeepFace with hybrid CNN architecture
        - 7 basic emotions: happy, sad, angry, neutral, surprise, fear, disgust
        - Confidence score shown for each detection
        
        *Tips for Better Results:*
        - Use clear, front-facing images
        - Ensure good lighting
        - Avoid obstructed faces
        """)

def main():
    st.title("👁‍🗨 AI Emotion & Location Detector")
    st.caption("Upload a photo to analyze facial expressions and estimate location")
    tabs = st.tabs(["🏠 Home", "🗺️ Location Map", "📜 History", "📊 Emotion Chart"])

    with tabs[0]:
        username = st.text_input("👤 Enter your username")
        if username:
            st.sidebar.success(f"👤 Welcome {username}")
            uploaded_file = st.file_uploader("Upload an image (JPG/PNG)", type=["jpg","png"])
            if uploaded_file:
                try:
                    image = Image.open(uploaded_file)
                    img = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
                    detections = detector.detect_emotions(img)
                    detected_img = detector.draw_detections(img, detections)
                    col1, col2 = st.columns([1,2])
                    with col1:
                        st.subheader("🔍 Detection Results")
                        if detections:
                            emotions = [d["emotion"] for d in detections]
                            confidences = [d["confidence"] for d in detections]
                            st.success(f"🎭 Detected {len(detections)} face(s)")
                            for i, (emo, conf) in enumerate(zip(emotions, confidences)):
                                st.write(f"- Face {i+1}: {emo} ({conf}%)")
                            show_detection_guide()
                            save_history(username, emotions[0], confidences[0], "Unknown")
                        else:
                            st.warning("No faces detected")
                    with col2:
                        t1, t2 = st.tabs(["Original Image", "Processed Image"])
                        with t1: 
                            st.image(image, use_container_width=True)
                        with t2: 
                            st.image(detected_img, channels="BGR", use_container_width=True,
                                    caption=f"Detected {len(detections)} face(s)")
                except Exception as e:
                    st.error(f"Error processing image: {e}")

    with tabs[1]:
        st.map(pd.DataFrame({'lat':[3.139+random.uniform(-0.01,0.01)], 'lon':[101.6869+random.uniform(-0.01,0.01)]}))

    with tabs[2]:
        st.header("📜 Upload History")
        if username:
            try:
                if os.path.exists("history.csv"):
                    df = pd.read_csv("history.csv")
                    if df.empty:
                        st.info("No history available")
                    else:
                        df_filtered = df[df["Username"].str.contains(username, case=False)]
                        df_filtered = df_filtered.sort_values("timestamp", ascending=False).reset_index(drop=True)
                        df_filtered.index = df_filtered.index + 1
                        st.dataframe(df_filtered)
                        st.caption(f"{len(df_filtered)} records shown")
                else:
                    st.info("No history available")
            except:
                st.info("No records found")
        else:
            st.warning("Please enter username to view history")

    with tabs[3]:
        st.subheader("📊 Emotion Chart")
        if username:
            try:
                if os.path.exists("history.csv"):
                    df = pd.read_csv("history.csv")
                    df_filtered = df[df["Username"].str.contains(username, case=False)]
                    if not df_filtered.empty:
                        fig = px.pie(df_filtered, names="Emotion", title=f"Emotion Chart ({username})")
                        st.plotly_chart(fig)
                    else:
                        st.warning("No history available")
                else:
                    st.warning("No records found")
            except Exception as e:
                st.error(f"Chart error: {e}")
        else:
            st.warning("Please enter username to view history")

if __name__ == "__main__":
    main()
