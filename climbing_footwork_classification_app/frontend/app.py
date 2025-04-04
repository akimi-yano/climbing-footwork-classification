import streamlit as st
import requests
import io
from PIL import Image

# Streamlit UI
st.title("Climbing Footwork Classification")

# Upload image
uploaded_file = st.file_uploader("Upload an image", type=["jpg", "png", "jpeg"])

# Model selection
# model_name = st.selectbox("Select a model", ["resnet50", "mobilenet_v3_large", "efficientnet_b2", "vgg16", "gpt_4_turbo"])
model_name = st.selectbox("Select a model", ["resnet50", "mobilenet_v3_large", "efficientnet_b2", "vgg16"])

if uploaded_file:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_container_width=True)

    # Convert image to bytes
    img_bytes = io.BytesIO()
    image.save(img_bytes, format="JPEG")
    img_bytes = img_bytes.getvalue()

    # API request
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
            if result['confidence'] is None:
                st.write(f"Prediction: **{class_mapping[result['label']]}** (Confidence: {result['confidence']:.2f})")
            else:
                st.write(f"Prediction: **{class_mapping[result['label']]}** (Confidence: {result['confidence']:.2f})")
        else:
            st.error("Error in prediction!")
