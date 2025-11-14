import streamlit as st
import cv2
import numpy as np
from PIL import Image

st.set_page_config(page_title="OpenCV Intro App", layout="wide")
st.title("📘 OpenCV – Image Processing Playground")
st.write("Upload an image and apply OpenCV transformations like in your HTML file.")

uploaded = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

if uploaded:
    # Read image
    img = Image.open(uploaded)
    img_np = np.array(img)

    st.subheader("🖼 Original Image")
    st.image(img_np, channels="RGB", use_column_width=True)

    # Convert to BGR for OpenCV operations
    bgr = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)

    # ------- BGR to RGB -------
    st.subheader("🔄 BGR → RGB Conversion")
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    st.image(rgb, channels="RGB")

    # ------- Grayscale -------
    st.subheader("⚫ Grayscale")
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    st.image(gray, clamp=True)

    # ------- Blur -------
    st.subheader("💨 Blurred Image")
    blurred = cv2.GaussianBlur(bgr, (15, 15), 0)
    st.image(cv2.cvtColor(blurred, cv2.COLOR_BGR2RGB))

    # ------- Canny edges -------
    st.subheader("⚡ Canny Edge Detection")
    edges = cv2.Canny(bgr, 100, 200)
    st.image(edges, clamp=True)

    # ------- Resize -------
    st.subheader("📏 Resize (50%)")
    resized = cv2.resize(bgr, (0, 0), fx=0.5, fy=0.5)
    st.image(cv2.cvtColor(resized, cv2.COLOR_BGR2RGB))

    # ------- Crop (center 200x200) -------
    st.subheader("✂ Cropped (200×200 from center)")
    h, w = bgr.shape[:2]
    ch, cw = h//2, w//2
    cropped = bgr[ch-100:ch+100, cw-100:cw+100]
    st.image(cv2.cvtColor(cropped, cv2.COLOR_BGR2RGB))

    # ------- Draw shapes -------
    st.subheader("🖍 Drawing Shapes")
    drawn = bgr.copy()
    cv2.rectangle(drawn, (50, 50), (250, 250), (0,255,0), 3)
    cv2.circle(drawn, (200, 200), 80, (255,0,0), 3)
    cv2.putText(drawn, "OpenCV!", (50, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)
    st.image(cv2.cvtColor(drawn, cv2.COLOR_BGR2RGB))
