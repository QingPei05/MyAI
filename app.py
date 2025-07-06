import cv2
import numpy as np
import streamlit as st
from PIL import Image, ImageDraw, ImageFont  # 添加缺失的导入

# 加载预训练模型
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')
smile_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_smile.xml')

def detect_emotion(img):
    """使用OpenCV检测情绪（快乐、平静、悲伤、愤怒）"""
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
        emotion = "平静"  # 默认
        
        # 愤怒判断
        if len(eyes) >= 2:
            eye_centers = [y + ey + eh/2 for (ex,ey,ew,eh) in eyes[:2]]
            avg_eye_height = np.mean(eye_centers)
            eye_sizes = [eh for (ex,ey,ew,eh) in eyes[:2]]
            avg_eye_size = np.mean(eye_sizes)
            
            if avg_eye_size > h/5 and avg_eye_height < h/2.5:
                emotion = "愤怒"
            elif avg_eye_height < h/3:
                emotion = "悲伤"
        
        # 快乐判断（优先判断）
        if len(smiles) > 0:
            emotion = "快乐"
        
        emotions.append(emotion)
    
    return emotions, faces

def draw_detections(img, emotions, faces):
    """确保中文标签正确显示的绘制函数"""
    output_img = img.copy()
    
    # 颜色映射
    color_map = {
        "快乐": (0, 255, 0),     # 绿色
        "平静": (255, 255, 0),   # 黄色
        "悲伤": (0, 0, 255),     # 红色
        "愤怒": (0, 165, 255)    # 橙色
    }
    
    for i, ((x,y,w,h), emotion) in enumerate(zip(faces, emotions)):
        color = color_map.get(emotion, (255, 255, 255))
        
        # 将OpenCV图像转为PIL图像
        pil_img = Image.fromarray(cv2.cvtColor(output_img, cv2.COLOR_BGR2RGB))
        draw = ImageDraw.Draw(pil_img)
        
        try:
            # 尝试加载中文字体
            font = ImageFont.truetype("simhei.ttf", 20)
        except:
            # 回退到默认字体
            font = ImageFont.load_default()
        
        # 绘制文本背景框
        text = emotion  # 直接使用中文标签
        text_width, text_height = draw.textsize(text, font=font)
        draw.rectangle(
            [(x, y - text_height - 10), (x + text_width + 10, y - 10)],
            fill=color,
            outline=color
        )
        
        # 绘制文本
        draw.text(
            (x + 5, y - text_height - 5),
            text,
            font=font,
            fill=(255, 255, 255)  # 白色文字
        
        # 转换回OpenCV格式
        output_img = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
        
        # 绘制人脸框
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
            
            # 使用两列布局
            col1, col2 = st.columns([1, 2])
            
            with col1:
                st.subheader("检测结果")
                if emotions:
                    emotion_count = {
                        "快乐": emotions.count("快乐"),
                        "平静": emotions.count("平静"),
                        "悲伤": emotions.count("悲伤"),
                        "愤怒": emotions.count("愤怒")
                    }
                    
                    result = []
                    for emo, cnt in emotion_count.items():
                        if cnt > 0:
                            result.append(f"{cnt}人{emo}")
                    st.success("，".join(result))
                    
                    st.markdown("---")
                    st.markdown("**检测原理**：")
                    st.write("""
                    - 😊 快乐: 检测到明显笑容
                    - 😠 愤怒: 眼睛睁大且位置偏高
                    - 😐 平静: 默认中性表情
                    - 😢 悲伤: 眼睛位置偏高
                    """)
                else:
                    st.warning("未检测到人脸")
            
            with col2:
                tab1, tab2 = st.tabs(["原始图片", "分析结果"])
                with tab1:
                    st.image(image, use_container_width=True)
                with tab2:
                    st.image(detected_img, channels="BGR", use_container_width=True,
                           caption=f"检测到 {len(faces)} 个人脸")
                
        except Exception as e:
            st.error(f"处理错误: {str(e)}")

if __name__ == "__main__":
    main()
