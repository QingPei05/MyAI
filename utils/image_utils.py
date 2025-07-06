import cv2
import numpy as np
from PIL import Image
import io

class ImageUtils:
    @staticmethod
    def upload_to_cv2(uploaded_file):
        """将Streamlit上传文件转为OpenCV格式"""
        image = Image.open(uploaded_file)
        return cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)

    @staticmethod
    def encode_cv2_to_bytes(img):
        """OpenCV图像转字节流"""
        _, img_encoded = cv2.imencode(".jpg", img)
        return img_encoded.tobytes()
