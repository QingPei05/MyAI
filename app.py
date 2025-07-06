import cv2
import numpy as np
import streamlit as st
from PIL import Image

# 加载预训练模型
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')
smile_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_smile.xml')

def detect_emotion(img):
    """增强版情绪检测（7种情绪）"""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.1, 7, minSize=(100, 100))  # 提高检测精度
    
    emotions = []
    for (x,y,w,h) in faces:
        roi_gray = gray[y:y+h, x:x+w]
        roi_color = img[y:y+h, x:x+w]
        
        # 检测面部特征（提高检测参数精度）
        smiles = smile_cascade.detectMultiScale(
            roi_gray, 
            scaleFactor=1.8, 
            minNeighbors=25,
            minSize=(25, 25)
        )
        eyes = eye_cascade.detectMultiScale(
            roi_gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(30, 30)
        )
        
        # 多特征情绪判断逻辑
        emotion = "平静"  # 默认
        
        # 眼睛特征分析
        eye_features = {"count": len(eyes), "positions": [], "sizes": []}
        for (ex,ey,ew,eh) in eyes:
            eye_features["positions"].append(ey)
            eye_features["sizes"].append(eh)
        
        # 愤怒/惊讶判断
        if eye_features["count"] >= 2:
            avg_eye_height = np.mean(eye_features["positions"])
            avg_eye_size = np.mean(eye_features["sizes"])
            
            if avg_eye_size > h/5:  # 大眼睛
                emotion = "惊讶" if avg_eye_height < h/3 else "愤怒"
            elif avg_eye_height > h/2.5:  # 眼睛位置低
                emotion = "悲伤"
        
        # 嘴巴特征分析
        if len(smiles) > 0:
            (sx,sy,sw,sh) = smiles[0]
            smile_ratio = sw / w  # 笑容相对宽度
            
            if smile_ratio > 0.4:
                emotion = "快乐"
            elif smile_ratio > 0.25 and eye_features["count"] >= 2:
                if np.mean(eye_features["positions"]) < h/3:
                    emotion = "羡慕"
        
        # 恐惧判断（眼睛紧张特征）
        if eye_features["count"] > 2 and np.mean(eye_features["sizes"]) < h/8:
            emotion = "恐惧"
        
        emotions.append(emotion)
    
    return emotions, faces

def draw_detections(img, emotions, faces):
    """在图像上绘制检测结果（7种情绪颜色标记）"""
    for (x,y,w,h), emotion in zip(faces, emotions):
        # 7种情绪的颜色映射
        color_map = {
            "快乐": (0, 255, 0),      # 绿色
            "平静": (255, 255, 0),    # 黄色
            "悲伤": (0, 0, 255),      # 红色
            "愤怒": (0, 100, 255),    # 橙色
            "惊讶": (255, 0, 255),    # 粉色
            "恐惧": (128, 0, 128),    # 紫色
            "羡慕": (64, 224, 208)    # 青绿色
        }
        color = color_map.get(emotion, (255,255,255))
        
        # 绘制人脸框和标签（保持大字体）
        cv2.rectangle(img, (x,y), (x+w,y+h), color, 3)
        cv2.putText(img, emotion, (x, y-15), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
    
    return img

def main():
    st.set_page_config(page_title="高级情绪检测系统", layout="wide")
    st.title("😊 高级情绪检测")
    
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
                # 情绪统计结果
                st.subheader("检测结果")
                if emotions:
                    emotion_count = {e: emotions.count(e) for e in set(emotions)}
                    
                    # 按固定顺序输出
                    emotion_order = ["快乐", "平静", "悲伤", "愤怒", "惊讶", "恐惧", "羡慕"]
                    result_parts = []
                    for e in emotion_order:
                        if e in emotion_count:
                            result_parts.append(f"{emotion_count[e]}人{e}")
                    
                    st.success("，".join(result_parts))
                    
                    # 添加准确度提示
                    st.markdown("---")
                    st.info("""
                    **准确度提示**：
                    - 正脸照片效果最佳
                    - 避免过度遮挡
                    - 保证足够光照
                    """)
                else:
                    st.warning("未检测到人脸")
            
            with col2:
                # 图片显示
                tab1, tab2 = st.tabs(["原始图片", "分析结果"])
                with tab1:
                    st.image(image, use_container_width=True)
                with tab2:
                    st.image(detected_img, channels="BGR", use_container_width=True)
                
        except Exception as e:
            st.error(f"处理错误: {str(e)}")

if __name__ == "__main__":
    main()
