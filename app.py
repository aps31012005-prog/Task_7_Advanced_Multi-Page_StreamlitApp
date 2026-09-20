import streamlit as st
import tensorflow as tf

# ==========================================
# Page Configuration & Styling
# ==========================================

st.set_page_config(
    page_title="CIFAR-10 Deep Learning App",
    page_icon="🧠",
    layout="wide"
)

# Complete Clean UI CSS (Hides GitHub, Top Header, Status, Fullscreen Buttons)
st.markdown(
    """
    <style>
    /* Metric Cards Styling */
    [data-testid="stMetricValue"] {
        font-size: 28px;
        font-weight: 700;
        color: #4CAF50;
    }
    
    /* Info Badge Style */
    .class-badge {
        background-color: #1E222D;
        border: 1px solid #313745;
        border-radius: 8px;
        padding: 10px;
        text-align: center;
        font-weight: 600;
        margin-bottom: 10px;
        color: #E0E0E0;
    }
    
    .class-badge:hover {
        border-color: #4CAF50;
        background-color: #262C3A;
    }

    /* Top Header, GitHub links, Fork button & Streamlit Menu Hide */
    [data-testid="stHeader"] {display: none !important;}
    #MainMenu {visibility: hidden !important;}
    footer {visibility: hidden !important;}
    header {visibility: hidden !important;}
    
    /* Toolbar elements, Fullscreen & Streamlit badges */
    .stApp > header {display: none !important;}
    button[title="View fullscreen"] {display: none !important;}
    [data-testid="stElementToolbar"] {display: none !important;}
    [data-testid="stStyledFullScreenButton"] {display: none !important;}
    [data-testid="stStatusWidget"] {display: none !important;}
    </style>
    """,
    unsafe_allow_html=True
)


# ==========================================
# Load Model & Globals
# ==========================================

class_names = [
    "Airplane ✈️", "Automobile 🚗", "Bird 🐦", "Cat 🐱", "Deer 🦌",
    "Dog 🐶", "Frog 🐸", "Horse 🐴", "Ship 🚢", "Truck 🚚"
]

@st.cache_resource
def load_model():
    return tf.keras.models.load_model("model/cifar10_cnn.keras")

try:
    model = load_model()
    model_loaded = True
except Exception:
    model_loaded = False


# ==========================================
# Sidebar Context
# ==========================================

st.sidebar.title("🧠 CIFAR-10 AI Platform")
st.sidebar.markdown("---")
st.sidebar.info(
    "**Deep Learning Showcase**\n\n"
    "This app classifies images across 10 categories using a custom CNN architecture."
)
st.sidebar.markdown("---")
st.sidebar.caption("⚡ Built with TensorFlow & Streamlit")


# ==========================================
# Hero Banner & Main Title
# ==========================================

st.title("🧠 CIFAR-10 Image Classification Suite")
st.caption("Interactive Neural Network Dashboard & Performance Analytics")

st.markdown(
    """
    > **Overview:** Experience real-time multi-class object recognition driven by a 
    Convolutional Neural Network (CNN) trained on the CIFAR-10 benchmark dataset.
    """
)

st.markdown("---")


# ==========================================
# Key Dataset & Performance Metrics
# ==========================================

st.subheader("📊 Key Overview Metrics")

m1, m2, m3, m4, m5 = st.columns(5)

with m1:
    st.metric("Training Set", "50,000", help="Number of images used during model training")
with m2:
    st.metric("Testing Set", "10,000", help="Independent images used for evaluation")
with m3:
    st.metric("Resolution", "32 × 32 px", help="RGB Color Input Dimensions")
with m4:
    st.metric("Target Classes", "10 Categories")
with m5:
    st.metric("Test Accuracy", "72.4%", delta="Epochs: 10")

st.markdown("---")


# ==========================================
# Class Categories Grid
# ==========================================

st.subheader("🏷️ Target Classification Categories")

cols = st.columns(5)

for idx, class_name in enumerate(class_names):
    with cols[idx % 5]:
        st.markdown(
            f'<div class="class-badge">Class {idx}<br><b>{class_name}</b></div>',
            unsafe_allow_html=True
        )

st.markdown("---")


# ==========================================
# Model Specs & Status
# ==========================================

st.subheader("🤖 Neural Network Configuration")

with st.expander("🔍 Click to view detailed model architecture parameters", expanded=True):
    col_a, col_b = st.columns(2)
    
    with col_a:
        st.markdown(
            """
            * **Architecture:** Sequential Convolutional Neural Network (CNN)
            * **Optimization Algorithm:** Adam
            * **Loss Calculation:** Sparse Categorical Crossentropy
            * **Classification Layer:** 10 Units with Softmax Activation
            """
        )
        
    with col_b:
        st.markdown(
            """
            * **Input Tensor Shape:** `(32, 32, 3)`
            * **Feature Extractors:** 2x Conv2D + MaxPooling Blocks
            * **Regularization:** Dropout (0.5 Rate)
            * **Hidden Layer Units:** 128 Dense Units (ReLU Activation)
            """
        )

# Final Status Callout
if model_loaded:
    st.success("✅ **Model Status:** Keras CNN pipeline loaded and ready for inference.")
else:
    st.error("⚠️ **Model Status:** Model file `model/cifar10_cnn.keras` not found. Please train or upload the weights.")

st.caption("👈 Use the sidebar navigation menu to jump directly to **Image Prediction** or **Model Performance**.")