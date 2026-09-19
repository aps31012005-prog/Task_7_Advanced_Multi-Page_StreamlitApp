import streamlit as st
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from tensorflow.keras.datasets import cifar10

# Page Configuration
st.set_page_config(page_title="Dataset Analytics", page_icon="📊", layout="wide")

st.title("📊 CIFAR-10 Dataset Analytics Dashboard")
st.markdown("Explore detailed visualizations, metrics, and distribution statistics of the CIFAR-10 dataset used for training.")

# Load CIFAR-10 Data
@st.cache_data
def load_cifar10_data():
    (x_train, y_train), (x_test, y_test) = cifar10.load_data()
    class_names = ['Airplane', 'Automobile', 'Bird', 'Cat', 'Deer', 'Dog', 'Frog', 'Horse', 'Ship', 'Truck']
    return x_train, y_train.flatten(), x_test, y_test.flatten(), class_names

x_train, y_train, x_test, y_test, class_names = load_cifar10_data()

# --- TOP METRICS / KPIS ---
st.header("📌 Key Dataset Metrics")
col1, col2, col3, col4, col5 = st.columns(5)
col1.metric("Total Samples", f"{len(x_train) + len(x_test):,}")
col2.metric("Training Samples", f"{len(x_train):,}")
col3.metric("Testing Samples", f"{len(x_test):,}")
col4.metric("Image Resolution", "32 x 32 x 3")
col5.metric("Total Classes", f"{len(class_names)}")

st.divider()

# --- VISUAL ROW 1: Class Distribution & Train/Test Split ---
st.header("📈 Data Distribution")
col_left, col_right = st.columns(2)

with col_left:
    st.subheader("Class Distribution (Balanced Dataset)")
    # Train Data Counting
    df_train_counts = pd.DataFrame(y_train, columns=['class']).value_counts().reset_index()
    df_train_counts.columns = ['class_id', 'count']
    df_train_counts['Class Name'] = [class_names[i] for i in df_train_counts['class_id']]
    
    fig_bar = px.bar(
        df_train_counts, 
        x='Class Name', 
        y='count', 
        color='Class Name',
        text='count',
        color_discrete_sequence=px.colors.qualitative.Pastel
    )
    fig_bar.update_layout(showlegend=False, height=380, margin=dict(l=20, r=20, t=30, b=20))
    st.plotly_chart(fig_bar, use_container_width=True)

with col_right:
    st.subheader("Train vs Test Dataset Split")
    df_split = pd.DataFrame({
        'Split': ['Training Set (83.3%)', 'Testing Set (16.7%)'],
        'Count': [len(x_train), len(x_test)]
    })
    fig_donut = px.pie(
        df_split, 
        values='Count', 
        names='Split', 
        hole=0.5,
        color_discrete_sequence=['#29b6f6', '#ab47bc']
    )
    fig_donut.update_layout(height=380, margin=dict(l=20, r=20, t=30, b=20))
    st.plotly_chart(fig_donut, use_container_width=True)

st.divider()

# --- VISUAL ROW 2: RGB Channel Intensity & Extra Visual 1 ---
st.header("🎨 Channel & Image Profile Analysis")
col_rgb, col_res = st.columns(2)

with col_rgb:
    st.subheader("RGB Pixel Channel Intensity Distribution")
    # Sample 1000 images for fast distribution rendering
    sample_imgs = x_train[:1000]
    r_channel = sample_imgs[:, :, :, 0].flatten()
    g_channel = sample_imgs[:, :, :, 1].flatten()
    b_channel = sample_imgs[:, :, :, 2].flatten()

    fig_rgb = go.Figure()
    fig_rgb.add_trace(go.Histogram(x=r_channel[:100000], name='Red Channel', marker_color='red', opacity=0.6))
    fig_rgb.add_trace(go.Histogram(x=g_channel[:100000], name='Green Channel', marker_color='green', opacity=0.6))
    fig_rgb.add_trace(go.Histogram(x=b_channel[:100000], name='Blue Channel', marker_color='blue', opacity=0.6))

    fig_rgb.update_layout(
        barmode='overlay',
        xaxis_title="Pixel Value (0-255)",
        yaxis_title="Frequency",
        height=380,
        margin=dict(l=20, r=20, t=30, b=20)
    )
    st.plotly_chart(fig_rgb, use_container_width=True)

with col_res:
    # EXTRA VISUAL 1: Class-wise Mean Image Brightness / Pixel Intensity Profile
    st.subheader("Extra Visual 1: Average Brightness Per Class")
    
    class_brightness = []
    for cls_idx in range(10):
        cls_imgs = x_train[y_train == cls_idx]
        mean_val = np.mean(cls_imgs) # Mean brightness across all channels
        class_brightness.append(mean_val)

    df_bright = pd.DataFrame({
        'Class': class_names,
        'Mean Intensity': class_brightness
    }).sort_values(by='Mean Intensity', ascending=True)

    fig_bright = px.bar(
        df_bright,
        x='Mean Intensity',
        y='Class',
        orientation='h',
        color='Mean Intensity',
        color_continuous_scale='Viridis',
        text_auto='.1f'
    )
    fig_bright.update_layout(height=380, margin=dict(l=20, r=20, t=30, b=20), coloraxis_showscale=False)
    st.plotly_chart(fig_bright, use_container_width=True)

st.divider()

# --- VISUAL ROW 3: Extra Visual 2 & Class Explorer ---
col_heatmap, col_explorer = st.columns([1, 1])

with col_heatmap:
    st.subheader("Extra Visual 2: Class-wise RGB Intensity Matrix")
    
    rgb_matrix = []
    for cls_idx in range(10):
        cls_imgs = x_train[y_train == cls_idx]
        r_mean = np.mean(cls_imgs[:, :, :, 0])
        g_mean = np.mean(cls_imgs[:, :, :, 1])
        b_mean = np.mean(cls_imgs[:, :, :, 2])
        rgb_matrix.append([r_mean, g_mean, b_mean])

    # Transpose matrix for clean wide landscape view
    df_heatmap = pd.DataFrame(rgb_matrix, index=class_names, columns=['Red Channel', 'Green Channel', 'Blue Channel']).T

    fig_heat = px.imshow(
        df_heatmap,
        labels=dict(x="CIFAR-10 Class Name", y="Color Channel", color="Avg Pixel Intensity"),
        x=class_names,
        y=['Red Channel', 'Green Channel', 'Blue Channel'],
        text_auto=".1f",
        color_continuous_scale='YlOrRd'
    )
    fig_heat.update_layout(
        height=320, 
        margin=dict(l=20, r=20, t=30, b=20),
        coloraxis_colorbar=dict(title="Intensity")
    )
    st.plotly_chart(fig_heat, use_container_width=True)

with col_explorer:
    st.subheader("🖼️ Dataset Sample Class Explorer")
    selected_class = st.selectbox("Select Class to Preview:", class_names)
    
    cls_index = class_names.index(selected_class)
    idx_list = np.where(y_train == cls_index)[0][:8]
    
    grid_cols = st.columns(4)
    for idx, col in zip(idx_list, grid_cols * 2):
        with col:
            st.image(x_train[idx], caption=f"{selected_class} #{idx}", use_container_width=True)

st.success("Dashboard successfully loaded! All dataset metrics and custom visual analysis are active.")