import cv2
import numpy as np
import streamlit as st
from PIL import Image

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
        
        # 愤怒判断（新增）
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
    """优化后的标签绘制函数"""
    output_img = img.copy()
    for i, ((x,y,w,h), emotion) in enumerate(zip(faces, emotions)):
        # 更醒目的颜色映射
        color_map = {
            "快乐": (0, 200, 0),     # 更亮的绿色
            "平静": (255, 255, 100),  # 更亮的黄色
            "悲伤": (200, 50, 50),    # 更柔和的红色
            "愤怒": (255, 150, 50)    # 更亮的橙色
        }
        color = color_map.get(emotion, (200, 200, 200))
        
        # 优化标签样式
        text = f"{emotion}"  # 移除了序号，只显示情绪
        font_scale = 0.9 if w > 100 else 0.7  # 根据人脸大小调整字体
        
        # 计算文本大小和位置
        (text_w, text_h), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, 2)
        
        # 绘制背景框（圆角矩形效果）
        cv2.rectangle(output_img, 
                     (x, y - text_h - 20), 
                     (x + text_w + 20, y - 10), 
                     color, -1)
        
        # 绘制文字（居中显示）
        cv2.putText(output_img, text, 
                   (x + 10, y - 15), 
                   cv2.FONT_HERSHEY_SIMPLEX, font_scale, 
                   (255, 255, 255), 2, cv2.LINE_AA)
        
        # 绘制人脸框（更粗的线条）
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
                        "快乐": "开心",
                        "平静": "平静",
                        "悲伤": "伤心",
                        "愤怒": "愤怒"
                    }
                    
                    emotion_count = {
                        "开心": emotions.count("快乐"),
                        "平静": emotions.count("平静"),
                        "伤心": emotions.count("悲伤"),
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
                    - 😊 开心: 检测到明显笑容
                    - 😠 愤怒: 眼睛睁大且位置偏高
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
                           caption=f"检测到 {len(faces)} 个人脸")  # 修改为只显示人脸数量
                
        except Exception as e:
            st.error(f"处理错误: {str(e)}")

if __name__ == "__main__":
    main()
