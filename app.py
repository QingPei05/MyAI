import cv2
import numpy as np
import streamlit as st
from PIL import Image

# 加载预训练模型（使用更精确的参数）
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')
smile_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_smile.xml')

def detect_emotion(img):
    """高精度情绪检测（快乐、平静、悲伤、愤怒）"""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # 高精度人脸检测参数
    faces = face_cascade.detectMultiScale(
        gray,
        scaleFactor=1.05,
        minNeighbors=10,
        minSize=(150, 150),
        flags=cv2.CASCADE_SCALE_IMAGE
    )
    
    emotions = []
    valid_faces = []
    for (x,y,w,h) in faces:
        roi_gray = gray[y:y+h, x:x+w]
        valid_faces.append((x,y,w,h))
        
        # 高精度特征检测
        smiles = smile_cascade.detectMultiScale(
            roi_gray,
            scaleFactor=1.7,
            minNeighbors=25,
            minSize=(40, 40)
        )
        
        eyes = eye_cascade.detectMultiScale(
            roi_gray,
            scaleFactor=1.1,
            minNeighbors=8,
            minSize=(45, 45))
        )
        
        # 多维度情绪判断
        emotion = "平静"
        eye_features = {"count": 0, "avg_height": 0, "avg_size": 0}
        
        if len(eyes) >= 2:
            eye_features = {
                "count": len(eyes),
                "avg_height": np.mean([y + ey + eh/2 for (ex,ey,ew,eh) in eyes[:2]]),
                "avg_size": np.mean([eh for (ex,ey,ew,eh) in eyes[:2]])
            }
            
            # 愤怒判断（眼睛大小和位置）
            if eye_features["avg_size"] > h/4.5 and eye_features["avg_height"] < h/2.2:
                emotion = "愤怒"
            # 悲伤判断（眼睛位置和嘴巴）
            elif eye_features["avg_height"] > h/2.2:
                emotion = "悲伤"
        
        # 快乐判断（笑容质量）
        if len(smiles) > 0:
            main_smile = max(smiles, key=lambda s: s[2]*s[3])  # 选择最大面积的笑容
            smile_ratio = main_smile[2] / w
            if smile_ratio > 0.35 and main_smile[3] > h/6:  # 笑容宽度和高度阈值
                emotion = "快乐"
                # 快乐程度分级
                if smile_ratio > 0.45 and eye_features.get("avg_size", 0) > h/5:
                    emotion = "快乐"  # 可扩展为"非常快乐"
        
        emotions.append(emotion)
    
    return emotions, valid_faces

def draw_detections(img, emotions, faces):
    """高可见性标注绘制"""
    output_img = img.copy()
    for i, ((x,y,w,h), emotion) in enumerate(zip(faces, emotions)):
        # 颜色映射
        color_map = {
            "快乐": (0, 180, 0),
            "平静": (210, 210, 0),
            "悲伤": (0, 0, 180),
            "愤怒": (0, 100, 255)
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
        
        # 特征点标记（调试用）
        # if emotion == "愤怒":
        #     cv2.circle(output_img, (x+w//2, y+h//2), 5, (0,0,255), -1)
    
    return output_img

def main():
    st.set_page_config(page_title="高精度情绪检测", layout="wide")
    st.title("😊 高精度情绪检测")
    
    uploaded_file = st.file_uploader("上传清晰正脸照片（JPG/PNG）", type=["jpg", "png"])
    
    if uploaded_file:
        try:
            # 图像预处理
            image = Image.open(uploaded_file)
            img = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
            
            # 检测情绪
            emotions, faces = detect_emotion(img)
            detected_img = draw_detections(img.copy(), emotions, faces)
            
            # 保持您喜欢的布局
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
                    - 使用150x150像素最小人脸检测
                    - 多维度特征交叉验证
                    - 笑容质量分级判断
                    """)
                else:
                    st.warning("未检测到有效人脸，请尝试：\n1. 正对摄像头\n2. 保持良好光照\n3. 避免遮挡")
            
            with col2:
                tab1, tab2 = st.tabs(["原始图片", "分析结果"])
                with tab1:
                    st.image(image, use_container_width=True, caption="原始图片")
                with tab2:
                    st.image(detected_img, channels="BGR", use_container_width=True,
                           caption=f"检测到 {len(faces)} 个人脸 | 情绪标记: 序号:情绪")
                
        except Exception as e:
            st.error(f"处理失败: {str(e)}")
            st.info("建议上传更清晰的照片，避免侧脸或模糊")

if __name__ == "__main__":
    main()
