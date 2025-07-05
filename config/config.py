# Google Cloud Settings
USE_GOOGLE_API = False  # Set to True if using Google Vision API
GOOGLE_CREDENTIALS = None  # Path to service account JSON or None

# OCR Settings
USE_EASYOCR = True
TESSERACT_PATH = None  # e.g., r'C:\Program Files\Tesseract-OCR\tesseract.exe'

# Emotion Detection
MIN_FACE_CONFIDENCE = 0.7
EMOTION_THRESHOLD = 0.2

# Video Processing
FRAME_SKIP = 5  # Analyze every 5th frame
