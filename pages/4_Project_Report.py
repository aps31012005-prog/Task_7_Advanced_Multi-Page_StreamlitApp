import streamlit as st

st.set_page_config(
    page_title="Project Report",
    page_icon="📋",
    layout="wide"
)

# Custom Styling for Clean Look
st.markdown(
    """
    <style>
    .report-card {
        background-color: #1E222D;
        border-left: 5px solid #4CAF50;
        padding: 15px;
        border-radius: 5px;
        margin-bottom: 20px;
    }
    </style>
    """,
    unsafe_allow_html=True
)

st.title("📋 Project Documentation & Comprehensive Report")
st.caption("Deep Learning Image Classification Pipeline & Web Deployment Specification")

st.markdown("---")

# ==========================================
# 1. Executive Summary
# ==========================================
st.header("1. Executive Summary")
st.markdown(
    """
    <div class="report-card">
    <b>Objective:</b> To design, train, and deploy an end-to-end Deep Learning application using TensorFlow/Keras 
    and Streamlit. The application performs real-time multi-class object recognition across 10 distinct visual classes 
    from the benchmark CIFAR-10 dataset.
    </div>
    """,
    unsafe_allow_html=True
)

# ==========================================
# 2. Dataset Overview & Preprocessing Workflow
# ==========================================
st.header("2. Dataset & Preprocessing Pipeline")
col1, col2 = st.columns(2)

with col1:
    st.subheader("📊 Dataset Statistics")
    st.markdown(
        """
        * **Dataset Name:** CIFAR-10 (Canadian Institute For Advanced Research)
        * **Total Images:** 60,000 RGB color images
        * **Split Ratio:** 50,000 Training | 10,000 Testing
        * **Image Dimensions:** 32 × 32 pixels × 3 color channels (RGB)
        * **Class Categories:** Airplane, Automobile, Bird, Cat, Deer, Dog, Frog, Horse, Ship, Truck
        """
    )

with col2:
    st.subheader("⚙️ Preprocessing Workflow")
    st.markdown(
        """
        1. **Feature Normalization:** Scaled pixel values from integer range `[0, 255]` to floating-point range `[0.0, 1.0]` by dividing by `255.0` to stabilize gradient descent.
        2. **Dimensionality Formatting:** Preserved `(32, 32, 3)` spatial input dimension for convolutional feature maps.
        3. **Batch Preprocessing:** Inference pipeline includes batch expansion (`np.expand_dims`) and automatic resizing to match input shapes.
        """
    )

st.markdown("---")

# ==========================================
# 3. Deep Learning Architecture
# ==========================================
st.header("3. CNN Model Architecture")

st.code(
    """
=================================================================
 Layer (type)                Output Shape              Param #   
=================================================================
 InputLayer                  (None, 32, 32, 3)         0         
 Conv2D (3x3, 32 filters)    (None, 30, 30, 32)        896       
 MaxPooling2D (2x2)          (None, 15, 15, 32)        0         
 Conv2D (3x3, 64 filters)    (None, 13, 13, 64)        18496     
 MaxPooling2D (2x2)          (None, 6, 6, 64)          0         
 Flatten                     (None, 2304)              0         
 Dense (ReLU)                (None, 128)               295040    
 Dropout (Rate = 0.5)        (None, 128)               0         
 Dense Output (Softmax)      (None, 10)                1290      
=================================================================
Total params: 315,722 | Trainable params: 315,722
=================================================================
    """,
    language="text"
)

# ==========================================
# 4. Training Setup & Hyperparameters
# ==========================================
st.header("4. Training Parameters & Configuration")

p1, p2, p3, p4 = st.columns(4)
p1.metric("Optimizer", "Adam (lr=0.001)")
p2.metric("Loss Function", "Sparse Crossentropy")
p3.metric("Batch Size", "64")
p4.metric("Epochs", "10")

st.markdown("---")

# ==========================================
# 5. Multi-Page Software Architecture
# ==========================================
st.header("5. Streamlit Application Architecture")
st.markdown(
    """
    The application follows a modular, multi-page structure enabled by Streamlit's native page router:
    
    * **`app.py` (Dashboard):** Provides dataset statistics, class listings, and high-level model specs.
    * **`Image_Prediction.py` (Inference Engine):** Handles file uploads (JPG, PNG), tensor preprocessing, model prediction, confidence scores, and probability bar charts.
    * **`Model_Performance.py` (Analytics Dashboard):** Displays accuracy/loss curves across training epochs, test evaluation metrics, and a confusion matrix.
    * **`Project_Report.py` (Documentation):** Houses the complete project documentation, system architecture, and observations.
    """
)

st.markdown("---")

# ==========================================
# 6. Key Observations & Recommendations
# ==========================================
st.header("6. Technical Observations & Future Enhancements")

st.markdown(
    """
    * **Impact of Normalization:** Scaling raw pixel values significantly reduced training loss oscillation during early epochs.
    * **Overfitting Control:** Adding a `0.5 Dropout` layer after the Dense 128 layer prevented the model from overfitting heavily to training set features.
    * **Resolution Constraints:** Standard 32×32 resolution can lose fine-grained features for complex images (e.g., distinguishing a Cat from a Dog).
    * **Recommended Upgrades:**
      1. **Data Augmentation:** Implement Random Flip, Rotation, and Zoom layers to improve generalizability.
      2. **Transfer Learning:** Upgrade base architecture to ResNet50 or MobileNetV2 pre-trained on ImageNet for >85% accuracy.
    """
)

st.success("✅ **Report Status:** Document finalized and verified for L&T Edutech LMS submission.")