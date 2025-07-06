import cv2
import numpy as np
import streamlit as st
from PIL import Image

# 加载预训练模型（使用更可靠的参数）
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')
smile_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_smile.xml')

def detect_emotion(img):
    """更可靠的情绪检测（快乐、平静、悲伤、愤怒）"""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # 优化的人脸检测参数
    faces = face_cascade.detectMultiScale(
        gray,
        scaleFactor=1.05,  # 更精细的缩放
        minNeighbors=8,    # 更高的邻居阈值
        minSize=(120, 120), # 最小人脸尺寸
        flags=cv2.CASCADE_SCALE_IMAGE
    )
    
    emotions = []
    valid_faces = []
    for (x,y,w,h) in faces:
        # 确保人脸区域有效
        if w < 50 or h < 50:  # 忽略过小的人脸
            continue
            
        roi_gray = gray[y:y+h, x:x+w]
        valid_faces.append((x,y,w,h))
        
        # 更可靠的微笑检测
        smiles = smile_cascade.detectMultiScale(
            roi_gray,
            scaleFactor=1.7,
            minNeighbors=22,
            minSize=(35, 35)
        )
        
        # 更可靠的眼睛检测
        eyes = eye_cascade.detectMultiScale(
            roi_gray,
            scaleFactor=1.05,
            minNeighbors=5,
            minSize=(40, 40)
        )
        
        # 增强的情绪判断逻辑
        emotion = "平静"  # 默认
        
        # 眼睛特征分析（确保有两只眼睛）
        if len(eyes) >= 2:
            eye_centers = [y + ey + eh/2 for (ex,ey,ew,eh) in eyes[:2]]  # 只取前两个眼睛
            avg_eye_height = sum(eye_centers) / len(eye_centers)
            
            # 愤怒判断（眼睛睁开程度）
            eye_openness = sum([eh for (ex,ey,ew,eh) in eyes[:2]]) / 2
            if eye_openness > h/5 and avg_eye_height < h/2.3:
                emotion = "愤怒"
            # 悲伤判断
            elif avg_eye_height > h/2.3:
                emotion = "悲伤"
        
        # 快乐判断（优先判断）
        if len(smiles) > 0:
            (sx,sy,sw,sh) = max(smiles, key=lambda s: s[2])  # 取最大笑容
            if sw > w/3:  # 笑容宽度阈值
                emotion = "快乐"
        
        emotions.append(emotion)
    
    return emotions, valid_faces

def draw_detections(img, emotions, faces):
    """确保标签显示的绘制方法"""
    output_img = img.copy()
    for (x,y,w,h), emotion in zip(faces, emotions):
        # 颜色映射
        color_map = {
            "快乐": (0, 200, 0),    # 更柔和的绿色
            "平静": (200, 200, 0),  # 更柔和的黄色
            "悲伤": (0, 0, 200),    # 更柔和的红色
            "愤怒": (0, 120, 255)   # 更醒目的橙色
        }
        color = color_map.get(emotion, (200,200,200))
        
        # 绘制带背景的文本（确保可见性）
        text_size = cv2.getTextSize(emotion, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)[0]
        cv2.rectangle(output_img, 
                     (x, y-35), 
                     (x + text_size[0] + 10, y-5), 
                     color, -1)  # 文本背景
        cv2.putText(output_img, emotion, 
                   (x+5, y-15), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, 
                   (255,255,255), 2)  # 白色文字
        
        # 绘制人脸框
        cv2.rectangle(output_img, (x,y), (x+w,y+h), color, 3)
    
    return output_img

def main():
    st.set_page_config(page_title="高精度情绪检测", layout="wide")
    st.title("😊 高精度情绪检测")
    
    uploaded_file = st.file_uploader("上传清晰正脸照片（JPG/PNG）", type=["jpg", "png"])
    
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
                    st.markdown("**优化说明**：")
                    st.write("""
                    - 采用更严格的人脸检测参数
                    - 情绪标签现在带有背景框
                    - 优化了愤怒和悲伤的判断逻辑
                    """)
                else:
                    st.warning("未检测到有效人脸，请上传更清晰的正脸照片")
            
            with col2:
                # 图片显示
                tab1, tab2 = st.tabs(["原始图片", "分析结果"])
                with tab1:
                    st.image(image, use_container_width=True, caption="上传的原始图片")
                with tab2:
                    st.image(detected_img, channels="BGR", use_container_width=True, 
                           caption=f"检测到 {len(faces)} 个人脸")
                
        except Exception as e:
            st.error(f"图片处理失败: {str(e)}")
            st.info("请尝试上传不同角度或更清晰的照片")

if __name__ == "__main__":
    main()
