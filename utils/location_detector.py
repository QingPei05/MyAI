import pytesseract
from geopy.geocoders import Nominatim
import re

def detect_location(image):
    # 使用OCR提取文字
    text = pytesseract.image_to_string(image)
    
    # 从文本中提取可能的地点信息
    locations = extract_location_keywords(text)
    
    if not locations:
        return "无法确定位置（请尝试更清晰的图片）"
    
    # 使用地理编码器获取详细地址
    geolocator = Nominatim(user_agent="location_emotion_app")
    try:
        location = geolocator.geocode(locations[0], addressdetails=True)
        if location:
            address = location.raw['address']
            country = address.get('country', '')
            state = address.get('state', '')
            city = address.get('city', address.get('town', ''))
            return f"{locations[0]}, {city}, {state}, {country}"
    except:
        pass
    
    return locations[0]

def extract_location_keywords(text):
    # 这里可以添加更多匹配规则
    patterns = [
        r'SMK\s\w+',  # 马来西亚学校
        r'Mount\s\w+',  # 山
        r'Jalan\s\w+',  # 马来西亚路名
        # 添加更多模式...
    ]
    
    found = []
    for pattern in patterns:
        matches = re.findall(pattern, text, re.IGNORECASE)
        found.extend(matches)
    
    return list(set(found))  # 去重
