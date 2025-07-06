def draw_detections(img, emotions, faces):
    """确保中文标签正确显示的绘制函数"""
    output_img = img.copy()
    
    # 中文字典映射
    emotion_dict = {
        "happy": "开心",
        "neutral": "平静",
        "sad": "伤心",
        "愤怒": "愤怒"  # 新增的愤怒情绪
    }
    
    for i, ((x,y,w,h), emotion) in enumerate(zip(faces, emotions)):
        # 颜色映射（使用更醒目的颜色）
        color_map = {
            "happy": (0, 255, 0),     # 绿色
            "neutral": (255, 255, 0), # 黄色
            "sad": (0, 0, 255),       # 红色
            "愤怒": (0, 165, 255)      # 橙色
        }
        color = color_map.get(emotion, (255, 255, 255))
        
        # 获取中文标签
        chinese_emotion = emotion_dict.get(emotion, emotion)
        
        # 设置字体（使用支持中文的字体）
        try:
            font = ImageFont.truetype("SimHei.ttf", 20)  # 黑体
        except:
            font = ImageFont.load_default()
        
        # 将OpenCV图像转为PIL图像
        pil_img = Image.fromarray(cv2.cvtColor(output_img, cv2.COLOR_BGR2RGB))
        draw = ImageDraw.Draw(pil_img)
        
        # 绘制文本背景框
        text_width, text_height = draw.textsize(chinese_emotion, font=font)
        draw.rectangle(
            [(x, y - text_height - 10), (x + text_width + 10, y - 10)],
            fill=tuple(color),
            outline=tuple(color)
        )
        
        # 绘制中文文本
        draw.text(
            (x + 5, y - text_height - 5),
            chinese_emotion,
            font=font,
            fill=(255, 255, 255)  # 白色文字
        )
        
        # 转换回OpenCV格式
        output_img = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
        
        # 绘制人脸框
        cv2.rectangle(output_img, (x,y), (x+w,y+h), color, 3)
    
    return output_img
