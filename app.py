import cv2
import numpy as np
import streamlit as st
from PIL import Image

# 加载预训练模型
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')
smile_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_smile.xml')

def detect_emotion(img):
    """使用OpenCV检测情绪（happy/neutral/sad）"""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    
    emotions = []
    for (x,y,w,h) in faces:
        roi_gray = gray[y:y+h, x:x+w]
        
        # 检测微笑
        smiles = smile_cascade.detectMultiScale(roi_gray, scaleFactor=1.8, minNeighbors=20)
        # 检测眼睛
        eyes = eye_cascade.detectMultiScale(roi_gray)
        
        # 情绪判断逻辑
        emotion = "neutral"  # 默认
        if len(smiles) > 0:
            emotion = "happy"
        elif len(eyes) > 0:
            if eyes[0][1] < h/3:  # 眼睛位置偏高
                emotion = "sad"
        
        emotions.append(emotion)
    
    return emotions, faces

def draw_detections(img, emotions, faces):
    """在图像上绘制检测结果"""
    for (x,y,w,h), emotion in zip(faces, emotions):
        # 人脸框颜色根据情绪变化
        color = {
            "happy": (0, 255, 0),    # 绿色
            "neutral": (255, 255, 0), # 黄色
            "sad": (0, 0, 255)       # 红色
        }.get(emotion, (255,255,255))
        
        cv2.rectangle(img, (x,y), (x+w,y+h), color, 2)
        cv2.putText(img, emotion, (x, y-10), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)
    
    return img

def main():
    st.set_page_config(page_title="情绪检测系统", layout="wide")
    st.title("😊 情绪检测")
    
    uploaded_file = st.file_uploader("上传图片（JPG/PNG）", type=["jpg", "png"])
    
    if uploaded_file:
        try:
            # 转换图片格式
            image = Image.open(uploaded_file)
            img = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
            
            # 检测情绪
            emotions, faces = detect_emotion(img)
            detected_img = draw_detections(img.copy(), emotions, faces)
            
            # 显示结果
            col1, col2 = st.columns(2)
            with col1:
                st.image(image, caption="原始图片", use_container_width=True)
            with col2:
                st.image(detected_img, channels="BGR", caption="分析结果", use_container_width=True)
            
            # 情绪统计输出
            st.subheader("检测结果")
            if emotions:
                # 统计每种情绪的人数
                emotion_count = {
                    "happy": emotions.count("happy"),
                    "neutral": emotions.count("neutral"),
                    "sad": emotions.count("sad")
                }
                
                # 生成简洁的输出文本
                result_parts = []
                if emotion_count["happy"] > 0:
                    result_parts.append(f"{emotion_count['happy']}人开心")
                if emotion_count["neutral"] > 0:
                    result_parts.append(f"{emotion_count['neutral']}人平静")
                if emotion_count["sad"] > 0:
                    result_parts.append(f"{emotion_count['sad']}人伤心")
                
                st.success("，".join(result_parts))
            else:
                st.warning("未检测到人脸")
                
        except Exception as e:
            st.error(f"处理错误: {str(e)}")

if __name__ == "__main__":
    main()
