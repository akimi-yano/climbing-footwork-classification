import streamlit as st
import requests
import io
from PIL import Image

st.set_page_config(layout="wide")

# Custom style for larger fonts
st.markdown("""
    <style>
    html, body, [class*="css"]  {
        font-size: 20px !important;
    }
    .stButton>button {
        font-size: 22px !important;
        padding: 10px 20px;
    }
    .stSelectbox label, .stFileUploader label {
        font-size: 22px !important;
    }
    .stSelectbox div[data-baseweb="select"] > div {
        font-size: 20px !important;
    }
    .stMarkdown h1 {
        font-size: 36px !important;
    }
    </style>
""", unsafe_allow_html=True)

# Streamlit UI
st.title("Climbing Footwork Classification")

col1, col2 = st.columns([2, 1])

with col1:
    uploaded_file = st.file_uploader("Upload an image", type=["jpg", "png", "jpeg"])

    model_name = st.selectbox(
        "Select a model",
        ["resnet50", "mobilenet_v3_large", "efficientnet_b2", "vgg16", "gpt_4_turbo"]
    )

    if uploaded_file:
        image = Image.open(uploaded_file)

        # Convert image to bytes
        img_bytes = io.BytesIO()
        image.save(img_bytes, format="JPEG")
        img_bytes = img_bytes.getvalue()

        if st.button("Classify"):
            endpoint = "https://climbing-image-classification-270454513285.us-central1.run.app"
            response = requests.post(f"{endpoint}/predict/{model_name}", files={"file": img_bytes})
            
            if response.status_code == 200:
                result = response.json()
                class_mapping = {
                    0: 'Heel Hook',
                    1: 'Toe Hook',
                    2: 'Others'
                }
                predicted_class = class_mapping[result['label']]

                st.markdown(
                    f"<div style='font-size:36px; font-weight:bold; color:#4CAF50;'>"
                    f"Prediction: {predicted_class} "
                    + (f"(Confidence: {result['confidence']:.2f})" if result['confidence'] > 0 else "") +
                    "</div>",
                    unsafe_allow_html=True
                )
            else:
                st.error("Error in prediction!")

with col2:
    if uploaded_file:
        st.image(image, caption="Uploaded Image", use_container_width=True)
