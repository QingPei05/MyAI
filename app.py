import cv2
import numpy as np
import streamlit as st
from PIL import Image

# 加载预训练模型
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')
smile_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_smile.xml')

def detect_emotion(img):
    """增强版情绪检测（9种情绪）"""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    
    emotions = []
    for (x,y,w,h) in faces:
        roi_gray = gray[y:y+h, x:x+w]
        roi_color = img[y:y+h, x:x+w]
        
        # 检测面部特征
        smiles = smile_cascade.detectMultiScale(roi_gray, scaleFactor=1.8, minNeighbors=20)
        eyes = eye_cascade.detectMultiScale(roi_gray, scaleFactor=1.1, minNeighbors=5)
        
        # 增强的情绪判断逻辑
        emotion = "平静"  # 默认
        
        # 眼睛特征分析
        eye_features = []
        if len(eyes) >= 2:  # 检测到两只眼睛
            eye1 = eyes[0]
            eye2 = eyes[1]
            eye_center_y = (eye1[1] + eye2[1]) / 2
            eye_openness = (eye1[3] + eye2[3]) / 2  # 眼睛高度
            
            # 愤怒/惊讶判断
            if eye_openness > h/6:
                emotion = "愤怒" if eye_center_y < h/3 else "惊讶"
            # 悲伤判断
            elif eye_center_y > h/2.5:
                emotion = "悲伤"
        
        # 嘴巴特征分析
        if len(smiles) > 0:
            (sx,sy,sw,sh) = smiles[0]
            smile_ratio = sw / w  # 笑容宽度与脸宽比例
            
            if smile_ratio > 0.4:
                emotion = "快乐"
            elif smile_ratio > 0.25:
                emotion = "骄傲" if eye_openness < h/8 else "快乐"
        
        # 特殊情绪判断
        if len(eyes) == 1:  # 单眼可见可能是厌恶
            emotion = "厌恶"
        elif len(eyes) > 2:  # 多眼检测可能是恐惧
            emotion = "恐惧"
        
        # 根据面部位置调整（羡慕通常有轻微抬头）
        if y < img.shape[0]/4 and emotion == "平静":
            emotion = "羡慕"
        
        emotions.append(emotion)
    
    return emotions, faces

def draw_detections(img, emotions, faces):
    """在图像上绘制检测结果（9种情绪颜色标记）"""
    for (x,y,w,h), emotion in zip(faces, emotions):
        # 9种情绪的颜色映射
        color_map = {
            "快乐": (0, 255, 0),      # 绿色
            "平静": (255, 255, 0),    # 黄色
            "悲伤": (0, 0, 255),      # 红色
            "愤怒": (0, 165, 255),    # 橙色
            "惊讶": (255, 0, 255),    # 粉色
            "恐惧": (128, 0, 128),    # 紫色
            "厌恶": (0, 128, 0),      # 深绿色
            "骄傲": (255, 215, 0),    # 金色
            "羡慕": (64, 224, 208)    # 青绿色
        }
        color = color_map.get(emotion, (255,255,255))
        
        # 绘制人脸框和标签
        cv2.rectangle(img, (x,y), (x+w,y+h), color, 3)
        cv2.putText(img, emotion, (x, y-15), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 3)
    
    return img

def main():
    st.set_page_config(page_title="高级情绪检测系统", layout="wide")
    st.title("😊😢😠 高级情绪检测")
    
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
                    
                    # 按情绪类型排序输出
                    emotion_order = ["快乐", "平静", "悲伤", "愤怒", 
                                    "惊讶", "恐惧", "厌恶", "骄傲", "羡慕"]
                    result_parts = []
                    for e in emotion_order:
                        if e in emotion_count and emotion_count[e] > 0:
                            result_parts.append(f"{emotion_count[e]}人{e}")
                    
                    st.success("，".join(result_parts))
                    
                    # 情绪说明
                    st.markdown("---")
                    st.markdown("**情绪说明**")
                    st.write("""
                    - 😊 快乐: 明显笑容
                    - 😐 平静: 中性表情
                    - 😢 悲伤: 眼睛下垂
                    - 😠 愤怒: 瞪大眼睛
                    - 😲 惊讶: 眼睛睁大
                    - 😨 恐惧: 眼睛紧张
                    - 🤢 厌恶: 单眼微闭
                    - 🦚 骄傲: 轻微微笑
                    - 😏 羡慕: 轻微抬头
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
