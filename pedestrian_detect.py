import cv2
import numpy as np
import os

# Set the path for the body classifier (Haar Cascade XML)
body_classifier_path = r"C:\Users\aashutosh\OneDrive\Attachments\Desktop\CNN_notes\13th- Haar cascade classifier\Haarcascades\haarcascade_fullbody.xml"

# Check if the classifier path exists
if not os.path.exists(body_classifier_path):
    print(f"Error: The classifier file does not exist at {body_classifier_path}")
    exit()

# Create the body classifier
body_classifier = cv2.CascadeClassifier(body_classifier_path)

# Check if the classifier is loaded successfully
if body_classifier.empty():
    print("Error: Could not load the body classifier. Make sure the XML file is valid and accessible.")
    exit()

# Set the path for the video file
video_path = r"C:\Users\aashutosh\OneDrive\Attachments\Desktop\training\17581212-uhd_3840_2160_60fps.mp4"

# Check if the video path exists
if not os.path.exists(video_path):
    print(f"Error: The video file does not exist at {video_path}")
    exit()

# Open the video file
cap = cv2.VideoCapture(video_path)

# Check if the video was opened successfully
if not cap.isOpened():
    print(f"Error: Could not open video file at {video_path}")
    exit()

print("Video opened successfully. Starting pedestrian detection...")

# Create a resizable window
cv2.namedWindow("Pedestrians", cv2.WINDOW_NORMAL)
cv2.resizeWindow("Pedestrians", 1280, 720)

while cap.isOpened():
    ret, frame = cap.read()

    if not ret:
        print("Video ended or failed to read frame. Exiting...")
        break

    # ↓↓↓ FIX: Resize frame to prevent zoom (4K → 720p)
    frame = cv2.resize(frame, (1280, 720))

    # Convert to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect bodies
    bodies = body_classifier.detectMultiScale(
        gray,
        scaleFactor=1.2,
        minNeighbors=3,
        minSize=(50, 50),
        flags=cv2.CASCADE_SCALE_IMAGE
    )

    # Draw bounding boxes
    for (x, y, w, h) in bodies:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 255), 2)

    # Show video
    cv2.imshow("Pedestrians", frame)

    # Press Enter to exit
    if cv2.waitKey(1) == 13:
        print("Exiting...")
        break

cap.release()
cv2.destroyAllWindows()
