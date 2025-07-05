def process_video(video_path, frame_skip=5):
    """使用生成器减少内存占用"""
    cap = cv2.VideoCapture(video_path)
    frame_count = 0
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
            
        if frame_count % frame_skip == 0:
            yield process_frame(frame)
            
        frame_count += 1
    
    cap.release()
