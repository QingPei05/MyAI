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
    """高质量标注绘制（带序号和背景框）"""
    output_img = img.copy()
    for i, ((x,y,w,h), emotion) in enumerate(zip(faces, emotions)):
        # 颜色映射
        color_map = {
            "happy": (0, 180, 0),    # 绿色
            "neutral": (210, 210, 0), # 黄色
            "sad": (0, 0, 180)       # 红色
        }
        color = color_map.get(emotion, (150,150,150))
        
        # 带背景的文本标签
        text = f"{i+1}:{emotion}"
        (text_w, text_h), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)
        cv2.rectangle(output_img, (x, y-40), (x+text_w+10, y-10), color, -1)
        cv2.putText(output_img, text, (x+5, y-20), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,255), 2)
        
        # 人脸框
        cv2.rectangle(output_img, (x,y), (x+w,y+h), color, 3)
    
    return output_img

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
                st.subheader("检测结果")
                if emotions:
                    # 中文字典映射
                    emotion_mapping = {
                        "happy": "开心",
                        "neutral": "平静",
                        "sad": "伤心"
                    }
                    
                    emotion_count = {
                        "开心": emotions.count("happy"),
                        "平静": emotions.count("neutral"),
                        "伤心": emotions.count("sad")
                    }
                    
                    result = []
                    for emo, cnt in emotion_count.items():
                        if cnt > 0:
                            result.append(f"{cnt}人{emo}")
                    st.success("，".join(result))
                    
                    st.markdown("---")
                    st.markdown("**检测原理**：")
                    st.write("""
                    - 😊 开心: 检测到明显笑容
                    - 😐 平静: 默认中性表情
                    - 😢 伤心: 眼睛位置偏高
                    """)
                else:
                    st.warning("未检测到人脸")
            
            with col2:
                # 使用选项卡显示图片
                tab1, tab2 = st.tabs(["原始图片", "分析结果"])
                with tab1:
                    st.image(image, use_container_width=True)
                with tab2:
                    st.image(detected_img, channels="BGR", use_container_width=True,
                           caption=f"检测到 {len(faces)} 个人脸 | 标记: 序号:情绪")
                
        except Exception as e:
            st.error(f"处理错误: {str(e)}")

if __name__ == "__main__":
    main()
