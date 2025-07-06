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
    
    results = []
    for (x,y,w,h) in faces:
        roi_gray = gray[y:y+h, x:x+w]
        roi_color = img[y:y+h, x:x+w]
        
        # 检测微笑
        smiles = smile_cascade.detectMultiScale(roi_gray, scaleFactor=1.8, minNeighbors=20)
        # 检测眼睛
        eyes = eye_cascade.detectMultiScale(roi_gray)
        
        # 情绪判断逻辑
        emotion = "neutral"  # 默认中性
        if len(smiles) > 0:
            emotion = "happy"
        elif len(eyes) > 0:
            if eyes[0][1] < h/3:  # 眼睛位置偏高
                emotion = "sad"
        
        results.append({
            "box": [x,y,w,h],
            "emotion": emotion,
            "landmarks": {
                "eyes": [(x+ex, y+ey, ew, eh) for (ex,ey,ew,eh) in eyes],
                "smiles": [(x+sx, y+sy, sw, sh) for (sx,sy,sw,sh) in smiles]
            }
        })
    
    return results

def draw_detections(img, results):
    """在图像上绘制检测结果"""
    for result in results:
        x,y,w,h = result["box"]
        
        # 绘制人脸框（颜色根据情绪变化）
        color = {
            "happy": (0, 255, 0),    # 绿色
            "neutral": (255, 255, 0), # 黄色
            "sad": (0, 0, 255)       # 红色
        }.get(result["emotion"], (255,255,255))
        
        cv2.rectangle(img, (x,y), (x+w,y+h), color, 2)
        
        # 标记情绪文本
        cv2.putText(img, result["emotion"], (x, y-10), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)
        
        # 绘制眼睛和微笑区域
        for (ex,ey,ew,eh) in result["landmarks"]["eyes"]:
            cv2.rectangle(img, (ex,ey), (ex+ew,ey+eh), (255,0,0), 1)
        for (sx,sy,sw,sh) in result["landmarks"]["smiles"]:
            cv2.rectangle(img, (sx,sy), (sx+sw,sy+sh), (0,255,255), 1)
    
    return img

def main():
    st.set_page_config(page_title="OpenCV情绪检测", layout="wide")
    st.title("😊 实时情绪分析")
    
    # 模式选择
    analysis_mode = st.radio(
        "选择输入模式",
        ["上传图片", "实时摄像头"],
        horizontal=True
    )
    
    if analysis_mode == "上传图片":
        uploaded_file = st.file_uploader("上传图片（JPG/PNG）", type=["jpg", "png"])
        if uploaded_file:
            try:
                # 转换图片格式
                image = Image.open(uploaded_file)
                img = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
                
                # 检测情绪
                results = detect_emotion(img)
                detected_img = draw_detections(img.copy(), results)
                
                # 显示结果
                col1, col2 = st.columns(2)
                with col1:
                    st.image(image, caption="原始图片", use_column_width=True)
                with col2:
                    st.image(detected_img, channels="BGR", caption="分析结果", use_column_width=True)
                
                # 文字结果
                for i, result in enumerate(results):
                    st.markdown(f"**人脸 {i+1}**:")
                    st.write(f"- 情绪: `{result['emotion']}`")
                    st.write(f"- 位置: `{result['box']}`")
                    
            except Exception as e:
                st.error(f"处理错误: {str(e)}")
    
    else:  # 实时摄像头模式
        st.warning("注意：摄像头功能需要本地运行或启用浏览器权限")
        run_camera = st.checkbox("启动摄像头")
        frame_placeholder = st.empty()
        
        if run_camera:
            cap = cv2.VideoCapture(0)
            stop_button = st.button("停止")
            
            while cap.isOpened() and not stop_button:
                ret, frame = cap.read()
                if not ret:
                    st.error("无法获取摄像头画面")
                    break
                
                # 实时分析
                results = detect_emotion(frame)
                detected_frame = draw_detections(frame, results)
                
                # 显示实时画面
                frame_placeholder.image(detected_frame, channels="BGR")
            
            cap.release()
            cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
