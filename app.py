from utils.api_client import AzureEmotionAPI
from utils.image_utils import ImageUtils
import streamlit as st

def main():
    st.set_page_config(page_title="实时情绪检测", layout="wide")
    st.title("🌍 Azure情绪识别")
    
    uploaded_file = st.file_uploader("上传人脸图片", type=["jpg", "png", "jpeg"])
    if uploaded_file:
        try:
            # 转换上传文件
            cv2_img = ImageUtils.upload_to_cv2(uploaded_file)
            img_bytes = ImageUtils.encode_cv2_to_bytes(cv2_img)
            
            # 调用API
            api = AzureEmotionAPI()
            results = api.analyze_emotion(img_bytes)
            
            # 显示结果
            col1, col2 = st.columns(2)
            with col1:
                st.image(uploaded_file, caption="原始图片", use_column_width=True)
            with col2:
                if results:
                    emotions = results[0]["faceAttributes"]["emotion"]
                    st.metric("主导情绪", max(emotions, key=emotions.get))
                    for emotion, score in emotions.items():
                        st.progress(score, text=f"{emotion}: {score:.2f}")
                else:
                    st.warning("未检测到有效人脸")
                    
        except Exception as e:
            st.error(f"分析失败: {str(e)}")

if __name__ == "__main__":
    main()
