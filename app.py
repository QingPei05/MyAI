def detect_emotion(img):
    """使用OpenCV检测情绪（返回英文标签）"""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    
    emotions = []
    for (x,y,w,h) in faces:
        roi_gray = gray[y:y+h, x:x+w]
        
        # 检测微笑
        smiles = smile_cascade.detectMultiScale(roi_gray, scaleFactor=1.8, minNeighbors=20)
        # 检测眼睛
        eyes = eye_cascade.detectMultiScale(roi_gray)
        
        # 情绪判断逻辑（英文标签）
        emotion = "neutral"  # 默认
        
        # 愤怒判断
        if len(eyes) >= 2:
            eye_centers = [y + ey + eh/2 for (ex,ey,ew,eh) in eyes[:2]]
            avg_eye_height = np.mean(eye_centers)
            eye_sizes = [eh for (ex,ey,ew,eh) in eyes[:2]]
            avg_eye_size = np.mean(eye_sizes)
            
            if avg_eye_size > h/5 and avg_eye_height < h/2.5:
                emotion = "angry"
            elif avg_eye_height < h/3:
                emotion = "sad"
        
        # 快乐判断
        if len(smiles) > 0:
            emotion = "happy"
        
        emotions.append(emotion)
    
    return emotions, faces

def draw_detections(img, emotions, faces):
    """绘制英文标签"""
    output_img = img.copy()
    
    # 颜色映射
    color_map = {
        "happy": (0, 255, 0),    # 绿色
        "neutral": (255, 255, 0), # 黄色
        "sad": (0, 0, 255),      # 红色
        "angry": (0, 165, 255)    # 橙色
    }
    
    for i, ((x,y,w,h), emotion) in enumerate(zip(faces, emotions)):
        color = color_map.get(emotion, (255, 255, 255))
        
        # 绘制人脸框
        cv2.rectangle(output_img, (x,y), (x+w,y+h), color, 3)
        
        # 添加英文标签
        cv2.putText(output_img, 
                   emotion.upper(), 
                   (x+5, y-10), 
                   cv2.FONT_HERSHEY_SIMPLEX, 
                   0.8, 
                   color, 
                   2)
    
    return output_img

def main():
    st.set_page_config(page_title="Emotion Detection", layout="wide")
    st.title("😊 Emotion Detection")
    
    uploaded_file = st.file_uploader("Upload Image (JPG/PNG)", type=["jpg", "png"])
    
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
                st.subheader("Detection Results")
                if emotions:
                    emotion_count = {
                        "Happy": emotions.count("happy"),
                        "Neutral": emotions.count("neutral"),
                        "Sad": emotions.count("sad"),
                        "Angry": emotions.count("angry")
                    }
                    
                    result = []
                    for emo, cnt in emotion_count.items():
                        if cnt > 0:
                            result.append(f"{cnt} {emo}")
                    st.success(", ".join(result))
                    
                    st.markdown("---")
                    st.markdown("**Detection Logic**:")
                    st.write("""
                    - 😊 Happy: Detected smile
                    - 😠 Angry: Wide open eyes in upper face
                    - 😐 Neutral: Default expression
                    - 😢 Sad: Eyes positioned higher
                    """)
                else:
                    st.warning("No faces detected")
            
            with col2:
                tab1, tab2 = st.tabs(["Original", "Analysis"])
                with tab1:
                    st.image(image, use_column_width=True)
                with tab2:
                    st.image(detected_img, channels="BGR", use_column_width=True,
                           caption=f"Detected {len(faces)} faces")
                
        except Exception as e:
            st.error(f"Processing error: {str(e)}")
