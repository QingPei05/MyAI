import cv2
import numpy as np
import streamlit as st
from PIL import Image

# 加载预训练模型（使用更精确的参数）
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')
smile_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_smile.xml')

def detect_emotion(img):
    """增强版情绪检测（快乐、平静、悲伤、愤怒）"""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # 提高人脸检测精度
    faces = face_cascade.detectMultiScale(
        gray,
        scaleFactor=1.1,  # 更精细的缩放
        minNeighbors=7,   # 更高的邻居阈值
        minSize=(100, 100) # 最小人脸尺寸
    )
    
    emotions = []
    for (x,y,w,h) in faces:
        roi_gray = gray[y:y+h, x:x+w]
        
        # 检测微笑（提高检测精度）
        smiles = smile_cascade.detectMultiScale(
            roi_gray,
            scaleFactor=1.8,
            minNeighbors=25,  # 更高的阈值减少误检
            minSize=(25, 25)
        )
        
        # 检测眼睛（提高检测精度）
        eyes = eye_cascade.detectMultiScale(
            roi_gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(30, 30)
        )
        
        # 增强的情绪判断逻辑
        emotion = "平静"  # 默认
        
        # 眼睛特征分析
        if len(eyes) >= 2:  # 确保检测到两只眼睛
            eye1, eye2 = eyes[0], eyes[1]
            eye_center_y = (eye1[1] + eye2[1]) / 2  # 眼睛中心平均高度
            eye_openness = (eye1[3] + eye2[3]) / 2  # 眼睛睁开程度
            
            # 愤怒判断（眼睛睁大且位置正常）
            if eye_openness > h/6 and eye_center_y < h/3:
                emotion = "愤怒"
            # 悲伤判断（眼睛位置偏低）
            elif eye_center_y > h/2.5:
                emotion = "悲伤"
        
        # 快乐判断（优先判断）
        if len(smiles) > 0:
            (sx,sy,sw,sh) = smiles[0]
            if sw/w > 0.3:  # 笑容宽度占脸宽比例
                emotion = "快乐"
        
        emotions.append(emotion)
    
    return emotions, faces

def draw_detections(img, emotions, faces):
    """在图像上绘制检测结果（4种情绪颜色标记）"""
    for (x,y,w,h), emotion in zip(faces, emotions):
        # 4种情绪的颜色映射
        color_map = {
            "快乐": (0, 255, 0),    # 绿色
            "平静": (255, 255, 0),  # 黄色
            "悲伤": (0, 0, 255),    # 红色
            "愤怒": (0, 100, 255)   # 橙色
        }
        color = color_map.get(emotion, (255,255,255))
        
        # 绘制人脸框和标签（保持大字体）
        cv2.rectangle(img, (x,y), (x+w,y+h), color, 3)
        cv2.putText(img, emotion, (x, y-15), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
    
    return img

def main():
    st.set_page_config(page_title="精准情绪检测", layout="wide")
    st.title("😊 精准情绪检测")
    
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
                    emotion_count = {
                        "开心": emotions.count("快乐"),
                        "平静": emotions.count("平静"),
                        "伤心": emotions.count("悲伤"),
                        "愤怒": emotions.count("愤怒")
                    }
                    
                    # 按固定顺序输出
                    result_parts = []
                    for emotion, count in emotion_count.items():
                        if count > 0:
                            result_parts.append(f"{count}人{emotion}")
                    
                    st.success("，".join(result_parts))
                    
                    # 添加准确度提示
                    st.markdown("---")
                    st.info("""
                    **准确度提示**：
                    - 正脸照片效果最佳
                    - 保持面部清晰可见
                    - 避免强烈侧光
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
