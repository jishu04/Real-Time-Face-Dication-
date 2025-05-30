import streamlit as st
import cv2
import numpy as np
from keras.models import model_from_json

# Load model only once
if "emotion_model" not in st.session_state:
    with open("emotiondetector.json", "r") as json_file:
        model_json = json_file.read()
    model = model_from_json(model_json)
    model.load_weights("emotiondetector.h5")
    st.session_state.emotion_model = model
else:
    model = st.session_state.emotion_model

# Initialize face detector
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

# Labels for emotions
labels = {
    0: 'angry', 1: 'disgust', 2: 'fear',
    3: 'happy', 4: 'neutral', 5: 'sad', 6: 'surprise'
}

# Feature extractor
def extract_features(image):
    image = np.array(image)
    image = image.reshape(1, 48, 48, 1)
    return image / 255.0

# UI Layout
st.title("😊 Real-Time Facial Emotion Detection")

# Camera state control
if "camera_on" not in st.session_state:
    st.session_state.camera_on = False

col1, col2 = st.columns([1, 1])
with col1:
    if st.button("▶️ Start Camera"):
        st.session_state.camera_on = True
with col2:
    if st.button("⏹️ Stop Camera"):
        st.session_state.camera_on = False

stframe = st.empty()

# Webcam logic
if st.session_state.camera_on:
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        st.error("❌ Could not access webcam.")
    else:
        while st.session_state.camera_on:
            ret, frame = cap.read()
            if not ret:
                st.warning("⚠️ Failed to capture video frame.")
                break

            frame = cv2.flip(frame, 1)
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(gray, 1.3, 5)

            for (x, y, w, h) in faces:
                roi_gray = gray[y:y + h, x:x + w]
                roi_gray = cv2.resize(roi_gray, (48, 48))
                features = extract_features(roi_gray)

                pred = model.predict(features)
                emotion_label = labels[np.argmax(pred)]

                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                cv2.putText(frame, emotion_label, (x, y - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)

            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            stframe.image(frame, channels="RGB")

        cap.release()
        stframe.empty()
