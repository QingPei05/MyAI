import cv2
import requests

class LocationModel:
    def __init__(self, api_key):
        self.api_key = api_key
        self.url = f"https://vision.googleapis.com/v1/images:annotate?key={self.api_key}"

    def detect_location(self, image):
        """
        检测图片中的地点。
        """
        # 将图像编码为 JPEG 格式并进行 API 调用
        _, encoded_image = cv2.imencode('.jpg', image)
        image_data = encoded_image.tobytes()

        # 构造请求数据
        request_data = {
            "requests": [
                {
                    "image": {
                        "content": image_data.decode('ISO-8859-1')  # 转换为字符串
                    },
                    "features": [
                        {
                            "type": "LANDMARK_DETECTION",
                            "maxResults": 1
                        }
                    ]
                }
            ]
        }

        # 发送请求并处理响应
        response = requests.post(self.url, json=request_data)
        if response.status_code == 200:
            landmarks = response.json().get("responses")[0].get("landmarkAnnotations")
            if landmarks:
                return landmarks[0]['description']  # 返回第一个检测到的地点
        return "未知地点"  # 默认返回
