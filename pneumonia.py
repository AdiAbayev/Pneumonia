import streamlit as st
from tensorflow.keras.models import load_model
import numpy as np
from PIL import Image

# --- Стили страницы ---
st.markdown("""
    <style>
        body {
            background: linear-gradient(to right, #eef2f3, #8e9eab);
            font-family: 'Arial', sans-serif;
            margin: 0;
            padding: 0;
            display: flex;
            justify-content: center;
            align-items: center;
            height: 100vh;
        }

        @keyframes marquee {
            0% { transform: translateX(100%); }
            100% { transform: translateX(-100%); }
        }

        .marquee-container {
            width: 100%;
            overflow: hidden;
            background: #007BFF;
            padding: 10px 0;
            border-radius: 8px;
            margin-bottom: 20px;
        }

        .marquee-text {
            display: inline-block;
            white-space: nowrap;
            color: white;
            font-size: 18px;
            font-weight: bold;
            animation: marquee 10s linear infinite;
        }

        .main-container {
            background: white;
            padding: 20px;
            border-radius: 12px;
            box-shadow: 2px 4px 10px rgba(0, 0, 0, 0.1);
            margin: 20px auto;
            width: 80%;  /* Изменено с 80% на 80% для адаптации на экранах */
            max-width: 1280px; /* Максимальная ширина */
            text-align: center;
        }

        .stButton>button {
            background: #007BFF;
            color: white;
            font-size: 18px;
            padding: 12px 20px;
            border-radius: 8px;
            border: none;
            transition: 0.3s;
        }
        .stButton>button:hover {
            background: #0056b3;
            transform: scale(1.05);
        }

        .stFileUploader>div {
            border: 2px dashed #007BFF;
            padding: 15px;
            border-radius: 10px;
        }

        .stImage>img {
            border-radius: 10px;
            box-shadow: 2px 2px 10px rgba(0, 0, 0, 0.2);
            max-width: 100%; /* Пропорциональное изображение */
            height: auto;
        }

        /* Для адаптации размера кнопки и других элементов */
        .stButton {
            display: inline-block;
            margin-top: 20px;
        }
    </style>
""", unsafe_allow_html=True)

# --- Загрузка модели ---
@st.cache_resource
def load_trained_model():
    model = load_model('pneumonia_detection_cnn_model.keras')
    return model

model = load_trained_model()

# --- Интерфейс ---
st.markdown('<div class="marquee-container"><span class="marquee-text">🚀 AI-Powered Pneumonia Detection - Upload Your X-Ray Now! 🔍</span></div>', unsafe_allow_html=True)

st.markdown('<div class="main-container">', unsafe_allow_html=True)
st.title("🔍 AI Pneumonia Diagnosis")
st.write("**Upload a chest X-ray, and our AI will detect pneumonia.**")

uploaded_file = st.file_uploader("📂 Upload an X-ray image", type=["png", "jpg", "jpeg"])

if uploaded_file:
    st.image(uploaded_file, caption="Uploaded Image", use_column_width=True)
    st.success("✅ Image successfully uploaded!")

    if st.button("🔍 Run Analysis"):
        with st.spinner("⚙️ AI is analyzing the image..."):
            # Обработка изображения
            img = Image.open(uploaded_file).convert('RGB')
            img = img.resize((224, 224))  # Подгоняем размер
            img_array = np.array(img) / 255.0  # Нормализация
            img_array = np.expand_dims(img_array, axis=0)  # Добавляем размер батча

            # Предсказание
            prediction = model.predict(img_array)[0][0]
            
            # Отображение результата
            import time
            time.sleep(1)

        # Вывод результата
        if prediction > 0.5:
            st.error(f"❗ Pneumonia detected with {prediction * 100:.2f}% probability.")
        else:
            st.success(f"✅ No pneumonia detected. Probability: {(1 - prediction) * 100:.2f}%.")
            
st.markdown('</div>', unsafe_allow_html=True)