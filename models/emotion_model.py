import cv2

class EmotionModel:
    def __init__(self):
        # 加载面部检测模型
        self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

    def detect_emotion(self, image):
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        faces = self.face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)

        emotions = {}
        for (x, y, w, h) in faces:
            # 假设每张脸检测到的情感都是“happy”
            emotions['happy'] = emotions.get('happy', 0) + 1

        return emotions if emotions else {"neutral": 1}  # 如果没有检测到人，返回中性情感
