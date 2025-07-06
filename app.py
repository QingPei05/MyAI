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
    """在图像上绘制检测结果（调整字体大小为12px左右）"""
    for (x,y,w,h), emotion in zip(faces, emotions):
        # 人脸框颜色根据情绪变化
        color = {
            "happy": (0, 255, 0),    # 绿色
            "neutral": (255, 255, 0), # 黄色
            "sad": (0, 0, 255)       # 红色
        }.get(emotion, (255,255,255))
        
        cv2.rectangle(img, (x,y), (x+w,y+h), color, 2)
        cv2.putText(img, emotion, (x, y-10), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)  # 调整为0.6对应约12px
    
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
            
            # 使用两列布局（左侧结果，右侧图片）
            col1, col2 = st.columns([1, 2])
            
            with col1:
                # 情绪统计结果
                st.subheader("检测结果")
                if emotions:
                    emotion_count = {
                        "开心": emotions.count("happy"),
                        "平静": emotions.count("neutral"),
                        "伤心": emotions.count("sad")
                    }
                    
                    result_parts = []
                    if emotion_count["开心"] > 0:
                        result_parts.append(f"{emotion_count['开心']}人开心")
                    if emotion_count["平静"] > 0:
                        result_parts.append(f"{emotion_count['平静']}人平静")
                    if emotion_count["伤心"] > 0:
                        result_parts.append(f"{emotion_count['伤心']}人伤心")
                    
                    st.success("，".join(result_parts))
                    
                    # 情绪分布图表
                    st.markdown("---")
                    st.markdown("**情绪分布**")
                    st.bar_chart(emotion_count)
                else:
                    st.warning("未检测到人脸")
            
            with col2:
                # 使用选项卡显示图片
                tab1, tab2 = st.tabs(["原始图片", "分析结果"])
                with tab1:
                    st.image(image, use_container_width=True)
                with tab2:
                    st.image(detected_img, channels="BGR", use_container_width=True)
                
        except Exception as e:
            st.error(f"处理错误: {str(e)}")

if __name__ == "__main__":
    main()
