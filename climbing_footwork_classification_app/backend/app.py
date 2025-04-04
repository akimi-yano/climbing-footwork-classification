import base64
import io
import os

from flask import Flask, request, jsonify
from PIL import Image
import numpy as np
import openai
import tensorflow as tf

app = Flask(__name__)

# Define model paths
model_paths = {
    "resnet50": "./models/resnet50.h5",
    "mobilenet_v3_large": "./models/mobilenet_v3_large.keras",
    "efficientnet_b2": "./models/efficientnet_b2.h5",
    "vgg16": "./models/vgg16.h5",
    "gpt_4_turbo": "",
}

models_cache = {}

def get_model(model_name):
    """Load the model if not already cached."""
    if model_name not in models_cache:
        try:
            models_cache[model_name] = tf.keras.models.load_model(model_paths[model_name])
            print(f"Loaded {model_name} model successfully.")
        except Exception as e:
            print(f"Error loading model {model_name}: {e}")
            return None
    return models_cache[model_name]

def preprocess_image(image):
    """Resize, normalize, and expand image dimensions for model inference."""
    image = image.resize((224, 224))  # Resize for model input
    image = np.array(image).astype(np.float32) / 255.0  # Normalize
    image = np.expand_dims(image, axis=0)  # Add batch dimension
    return image

def encode_image_base64(image):
    """
    Converts a PIL image to a base64-encoded string.
    """
    img_byte_array = io.BytesIO()
    image.save(img_byte_array, format="JPEG")
    img_byte_array.seek(0)  # Move pointer to the start
    return base64.b64encode(img_byte_array.read()).decode('utf-8')

def classify_with_gpt(image):
    """
    Classifies an image using GPT-4V.
    Uses base64 encoding instead of an online URL.
    Returns an integer: 0 (heel_hook), 1 (toe_hook), or 2 (others).
    """

    # Preprocess the image for classification
    processed_image = preprocess_image(image)

    # Encode image as base64
    encoded_image = encode_image_base64(image)

    # Define prompt & messages
    messages = [
        {"role": "user", "content": [
            {"type": "text", "text": """
            You are an expert in rock climbing techniques. Classify the given image into one of these categories:
            - 0 (heel_hook): If the climber's **heel** is hooked onto a hold.
            - 1 (toe_hook): If the climber's **toe** is hooked onto a hold.
            - 2 (others): If neither technique is present.

            **Respond with only a single integer (0, 1, or 2). Do not include any explanation or text.**
            """},
            {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{encoded_image}"}}
        ]}
    ]

    client = openai.OpenAI()

    # API call to classify a single image
    response = client.chat.completions.create(
        model="gpt-4-turbo",
        messages=messages,
        max_tokens=1  # Limit response to one token (single integer)
    )

    # Extract and return integer prediction
    prediction = response.choices[0].message.content.strip()

    try:
        return int(prediction)  # Convert to integer
    except ValueError:
        return 3  # Default value if response is unexpected

@app.route("/predict/<model_name>", methods=["POST"])
def predict(model_name):
    if model_name not in model_paths:
        return jsonify({"error": "Model not found"}), 404

    if model_name == 'gpt_4_turbo':
        try:
            image = Image.open(io.BytesIO(file.read())).convert("RGB")
            pred = classify_with_gpt(image)
            # print(pred)
            confidence = None
            return jsonify({"label": pred, "confidence": confidence})
        except Exception as e:
            # print(e)
            return jsonify({"error": f"Prediction error: {str(e)}"}), 500
    else:
        model = get_model(model_name)
        if model is None:
            return jsonify({"error": f"Failed to load model {model_name}"}), 500

        file = request.files.get("file")
        if not file:
            return jsonify({"error": "No file provided"}), 400

        try:
            image = Image.open(io.BytesIO(file.read())).convert("RGB")
            input_tensor = preprocess_image(image)

            # Perform inference
            predictions = model.predict(input_tensor)
            label = int(np.argmax(predictions))
            confidence = float(np.max(predictions))

            return jsonify({"label": label, "confidence": confidence})

        except Exception as e:
            return jsonify({"error": f"Prediction error: {str(e)}"}), 500


if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=int(os.environ.get("PORT", 8080)))
