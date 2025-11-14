import cv2
import tempfile
import streamlit as st
import numpy as np
from pathlib import Path

st.title("🚗👨‍🦰 Object Detection from Video (Cars + Pedestrians)")
st.write("Upload a video and choose detection type.")

# -----------------------------
# Sidebar options
# -----------------------------
option = st.sidebar.selectbox(
    "Select Detection Type",
    ["Pedestrian Detection", "Car Detection"]
)

if option == "Pedestrian Detection":
    cascade_path = r"C:\Users\aashutosh\OneDrive\Attachments\Desktop\CNN_notes\12th,  - Intro to cv2\opencv\Haarcascades\haarcascade_fullbody.xml"
else:
    cascade_path = r"C:\Users\aashutosh\OneDrive\Attachments\Desktop\CNN_notes\12th,  - Intro to cv2\opencv\Haarcascades\haarcascade_car.xml"

classifier = cv2.CascadeClassifier(cascade_path)

# -----------------------------
# Upload video
# -----------------------------
uploaded_file = st.file_uploader("Upload a video file", type=["mp4", "avi", "mov", "mkv"])

if uploaded_file:

    # Save video temporarily
    tfile = tempfile.NamedTemporaryFile(delete=False)
    tfile.write(uploaded_file.read())
    video_path = tfile.name

    stframe = st.empty()

    cap = cv2.VideoCapture(video_path)

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Resize for stable stream
        frame = cv2.resize(frame, (800, 500))

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Detection
        objects = classifier.detectMultiScale(gray, 1.2, 3)

        for (x, y, w, h) in objects:
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 255), 2)

        # Convert to streamlit-friendly format
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        stframe.image(frame, channels="RGB")

    cap.release()

