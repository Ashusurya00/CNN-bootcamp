import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

st.set_page_config(page_title="Image Transformations", layout="wide")

st.title("📸 Image Transformations using NumPy & Matplotlib")
st.write("Upload an image and view colorful transformations like your HTML file.")

uploaded_image = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

if uploaded_image:
    img = Image.open(uploaded_image)
    img_np = np.array(img)

    st.subheader("🖼 Original Image")
    st.image(img, use_column_width=True)

    # Show shape
    st.write(f"Image shape: {img_np.shape}")

    # RGB Channels
    R = img_np[:, :, 0]
    G = img_np[:, :, 1]
    B = img_np[:, :, 2]

    # --- Display RGB Channels ---
    st.subheader("🎨 RGB Channels")
    col1, col2, col3 = st.columns(3)

    with col1:
        st.write("🔴 **Red Channel**")
        st.image(R, clamp=True)

    with col2:
        st.write("🟢 **Green Channel**")
        st.image(G, clamp=True)

    with col3:
        st.write("🔵 **Blue Channel**")
        st.image(B, clamp=True)

    # --- Grayscale ---
    st.subheader("⚫ Grayscale Image")
    gray = np.mean(img_np, axis=2).astype(np.uint8)
    st.image(gray, clamp=True)

    # --- Flattened Image ---
    st.subheader("📏 Flattened Image Array")
    flat = img_np.flatten()
    st.code(f"Flattened array shape: {flat.shape}")

    # --- Transposed Image ---
    st.subheader("🔄 Transposed Image")
    transposed = np.transpose(img_np, (1, 0, 2))
    st.image(transposed, use_column_width=True)

    # --- Reshaped Pixel Map ---
    st.subheader("🧩 Pixel Map (10x10 Example)")
    try:
        small = img_np[:10, :10, :]
        st.code(small)
    except:
        st.warning("Image is too small for 10x10 sample.")
