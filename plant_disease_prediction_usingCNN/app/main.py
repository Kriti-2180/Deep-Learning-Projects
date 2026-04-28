import streamlit as st
import numpy as np
from PIL import Image
import json
from tensorflow.keras.models import load_model

# Load class indices

with open("class_indices.json", "r") as f:
    class_indices = json.load(f)

# Reverse mapping

class_labels = {v: k for k, v in class_indices.items()}

# Load model (.keras format)

model = load_model("trained_models/plant_model.h5", compile=False)

# App UI

st.title("🌿 Plant Disease Prediction")

uploaded_file = st.file_uploader("Upload a leaf image", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).resize((224, 224))
    st.image(image, caption="Uploaded Image", use_column_width=True)

# Preprocess image

img_array = np.array(image) / 255.0
img_array = np.expand_dims(img_array, axis=0)

if st.button("Predict"):
    predictions = model.predict(img_array)
    predicted_class = np.argmax(predictions, axis=1)[0]
    label = class_labels.get(predicted_class, "Unknown")

    st.success(f"🌱 Prediction: {label}")

