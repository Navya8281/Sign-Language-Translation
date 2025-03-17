import streamlit as st
import cv2
import numpy as np
import mediapipe as mp
import tensorflow as tf

# Apply Custom CSS for Styling
st.markdown("""
    <style>
    /* Main title styling */
    .stTitle {
        font-size: 36px;
        font-weight: bold;
        color: white;
        text-align: center;
        margin-bottom: 10px;
    }

    /* Subtitle styling */
    .stSubtitle {
        font-size: 18px;
        color: white;
        text-align: center;
        margin-bottom: 20px;
    }

    /* Button styling */
    .stButton>button {
        width: 100px;
        height: 40px;
        font-size: 16px;
        font-weight: bold;
        border-radius: 8px;
        margin: 10px auto;
        display: block;
        color: white;
        background-color: red !important;
    }

    /* Webcam feed container */
    .webcam-container {
        border: 2px solid #2E86C1;
        border-radius: 10px;
        padding: 10px;
        margin-top: 20px;
        margin-bottom: 20px;
    }

    /* Predicted sign styling */
    .predicted-sign {
        font-size: 24px;
        font-weight: bold;
        color: white;
        text-align: center;
        margin-top: 20px;
    }

    /* Footer styling */
    .footer {
        font-size: 14px;
        color: #666;
        text-align: center;
        margin-top: 30px;
    }
    </style>
""", unsafe_allow_html=True)

# Define sign actions
actions = np.array(['hello', 'thanks', 'iloveyou'])

# Initialize Mediapipe models
mp_holistic = mp.solutions.holistic  
mp_drawing = mp.solutions.drawing_utils  

# Function for Mediapipe detection
def mediapipe_detection(image, model):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image.flags.writeable = False
    results = model.process(image)
    image.flags.writeable = True
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    return image, results

# Extract keypoints from Mediapipe results
def extract_keypoints(results):
    pose = np.array([[res.x, res.y, res.z, res.visibility] for res in results.pose_landmarks.landmark]).flatten() if results.pose_landmarks else np.zeros(33*4)
    face = np.array([[res.x, res.y, res.z] for res in results.face_landmarks.landmark]).flatten() if results.face_landmarks else np.zeros(468*3)
    lh = np.array([[res.x, res.y, res.z] for res in results.left_hand_landmarks.landmark]).flatten() if results.left_hand_landmarks else np.zeros(21*3)
    rh = np.array([[res.x, res.y, res.z] for res in results.right_hand_landmarks.landmark]).flatten() if results.right_hand_landmarks else np.zeros(21*3)
    return np.concatenate([pose, face, lh, rh]), lh, rh

# Load trained model
model = tf.keras.models.load_model('model.h5')
model.load_weights('model_weights.h5')

# Streamlit App
st.markdown("<h1 class='stTitle'>🤟 Real-Time Sign Language Detection</h1>", unsafe_allow_html=True)
st.markdown("<h3 class='stSubtitle'>📸 Live Camera Feed</h3>", unsafe_allow_html=True)
st.markdown("<p class='stSubtitle'>Start the webcam to detect sign language gestures in real-time.</p>", unsafe_allow_html=True)

# Placeholder for the webcam feed
FRAME_WINDOW = st.empty()

# Placeholder for the predicted sign
predicted_sign = st.empty()

# Initialize session state for camera control
if "camera_active" not in st.session_state:
    st.session_state.camera_active = False

# Button Logic
if not st.session_state.camera_active:
    if st.button("START"):
        st.session_state.camera_active = True
        st.rerun()  # Rerun the app to update the UI
else:
    if st.button("STOP"):
        st.session_state.camera_active = False
        FRAME_WINDOW.empty()  # Clear webcam feed
        predicted_sign.empty()  # Clear predicted sign
        st.rerun()  # Rerun the app to update the UI

# Open webcam
cap = cv2.VideoCapture(0)

# Set Mediapipe model
with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
    sequence = []
    threshold = 0.5

    while st.session_state.camera_active:
        ret, frame = cap.read()
        if not ret:
            st.error("Failed to capture image from webcam.")
            break

        image, results = mediapipe_detection(frame, holistic)

        # Draw landmarks
        mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS)
        mp_drawing.draw_landmarks(image, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS)
        mp_drawing.draw_landmarks(image, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS)

        # Process keypoints
        keypoints, left_hand, right_hand = extract_keypoints(results)
        sequence.append(keypoints)
        sequence = sequence[-30:]

        # Check if at least one hand is detected
        if len(sequence) == 30 and (np.any(left_hand) or np.any(right_hand)):
            res = model.predict(np.expand_dims(sequence, axis=0))[0]
            predicted_word = actions[np.argmax(res)] if res[np.argmax(res)] > threshold else ""
        else:
            predicted_word = ""

        # Display prediction only if a hand is detected
        if predicted_word:
            cv2.putText(image, predicted_word, (10, 30), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)

        # Convert the image from BGR to RGB
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        FRAME_WINDOW.image(image)

        # Display the predicted sign only if the video feed is active
        predicted_sign.markdown(f"<h3 class='predicted-sign'>Predicted Sign: {predicted_word}</h3>", unsafe_allow_html=True)

    cap.release()

