import cv2
import numpy as np
import streamlit as st
from PIL import Image

# 加载预训练的人脸和笑脸检测器
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
smile_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_smile.xml')

def detect_emotion(img):
    """使用OpenCV检测笑脸"""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    
    results = []
    for (x,y,w,h) in faces:
        roi_gray = gray[y:y+h, x:x+w]
        smiles = smile_cascade.detectMultiScale(roi_gray, scaleFactor=1.8, minNeighbors=20)
        
        # 如果有笑脸则标记为"happy"，否则为"neutral"
        emotion = "happy" if len(smiles) > 0 else "neutral"
        results.append({
            "box": [x,y,w,h],
            "emotion": emotion
        })
    
    return results

def main():
    st.set_page_config(page_title="OpenCV情绪检测", layout="wide")
    st.title("😊 纯OpenCV情绪分析")
    
    uploaded_file = st.file_uploader("上传图片（支持JPG/PNG）", type=["jpg", "png"])
    if uploaded_file:
        try:
            # 转换图片格式
            image = Image.open(uploaded_file)
            img = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
            
            # 检测情绪
            results = detect_emotion(img)
            
            # 显示结果
            col1, col2 = st.columns(2)
            with col1:
                st.image(image, caption="原始图片", use_column_width=True)
            
            with col2:
                if len(results) > 0:
                    for i, result in enumerate(results):
                        st.markdown(f"**人脸 {i+1}**:")
                        st.write(f"- 情绪: {result['emotion']}")
                        st.write(f"- 位置: {result['box']}")
                else:
                    st.warning("未检测到人脸")
                    
        except Exception as e:
            st.error(f"错误: {str(e)}")

if __name__ == "__main__":
    main()
