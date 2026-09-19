import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image


# ==========================================
# Page Configuration
# ==========================================

st.set_page_config(
    page_title="Image Prediction",
    page_icon="🔮",
    layout="wide"
)


# ==========================================
# CIFAR-10 Class Names
# ==========================================

class_names = [
    "Airplane",
    "Automobile",
    "Bird",
    "Cat",
    "Deer",
    "Dog",
    "Frog",
    "Horse",
    "Ship",
    "Truck"
]


# ==========================================
# Load Trained Model
# ==========================================

@st.cache_resource
def load_model():

    return tf.keras.models.load_model(
        "model/cifar10_cnn.keras"
    )


model = load_model()


# ==========================================
# Page Title
# ==========================================

st.title("🔮 CIFAR-10 Image Prediction")

st.write(
    "Upload an image and the trained CNN model will "
    "predict its CIFAR-10 class."
)


# ==========================================
# File Uploader
# ==========================================

uploaded_file = st.file_uploader(
    "📤 Upload an image",
    type=["jpg", "jpeg", "png"]
)


# ==========================================
# Prediction
# ==========================================

if uploaded_file is not None:

    image = Image.open(uploaded_file).convert("RGB")

    # Display uploaded image
    col1, col2 = st.columns(2)

    with col1:

        st.subheader("Uploaded Image")

        st.image(
            image,
            caption="Input Image",
            width=300
        )

    # Resize image to CIFAR-10 input size
    resized_image = image.resize((32, 32))

    # Convert image to NumPy array
    image_array = np.array(resized_image)

    # Normalize pixel values
    image_array = image_array.astype("float32") / 255.0

    # Add batch dimension
    image_array = np.expand_dims(
        image_array,
        axis=0
    )

    # Make prediction
    predictions = model.predict(
        image_array,
        verbose=0
    )

    predicted_index = np.argmax(predictions[0])

    predicted_class = class_names[predicted_index]

    confidence = predictions[0][predicted_index] * 100


    # ==========================================
    # Display Prediction
    # ==========================================

    with col2:

        st.subheader("🤖 Prediction")

        st.success(
            f"Predicted Class: **{predicted_class}**"
        )

        st.metric(
            "Confidence",
            f"{confidence:.2f}%"
        )


    # ==========================================
    # Prediction Probabilities
    # ==========================================

    st.markdown("---")

    st.subheader("📊 Prediction Probabilities")

    probability_data = {
        class_names[i]: float(predictions[0][i])
        for i in range(10)
    }

    st.bar_chart(probability_data)

else:

    st.info(
        "👆 Please upload a JPG, JPEG, or PNG image to begin."
    )