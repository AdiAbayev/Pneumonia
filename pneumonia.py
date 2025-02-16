import streamlit as st
import tensorflow as tf
from tensorflow.keras.models import load_model
from PIL import Image
import numpy as np

model = load_model('pneumonia_detection_cnn_model.keras')

def preprocess_image(image):
    image = image.resize((224, 224))
    image = np.array(image) / 255.0
    if len(image.shape) == 2:
        image = np.stack((image,)*3, axis=-1)
    image = np.expand_dims(image, axis=0)
    return image

st.title("Pneumonia Detection App")
st.write("Upload an X-ray of the lungs for diagnosis")

uploaded_file = st.file_uploader("Select an image", type=["jpg", "jpeg", "png"])
if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded image', use_container_width=True)

    st.write("Image processing...")
    img = preprocess_image(image)

    prediction = model.predict(img)
    result = "Pneumonia has been detected" if prediction[0][0] > 0.5 else "Pneumonia has not been detected"

    st.write(f"**Result:** {result}")