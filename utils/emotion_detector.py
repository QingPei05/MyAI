from deepface import DeepFace
import cv2

def detect_emotions(image):
    try:
        # 转换图像为RGB（如果必要）
        if len(image.shape) == 2:  # 灰度图
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        elif image.shape[2] == 4:  # RGBA
            image = image[:, :, :3]
            
        # 使用DeepFace分析情绪
        analysis = DeepFace.analyze(img_path=image, actions=['emotion'], enforce_detection=False)
        
        if not isinstance(analysis, list):
            analysis = [analysis]
            
        results = {}
        for face in analysis:
            emotion = face['dominant_emotion']
            results[emotion] = results.get(emotion, 0) + 1
            
        if not results:
            return "未检测到人脸"
            
        return ", ".join([f"{count} {emotion}" for emotion, count in results.items()])
        
    except Exception as e:
        return f"情绪分析错误: {str(e)}"
