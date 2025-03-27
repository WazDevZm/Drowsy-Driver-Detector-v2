#import the system from th agile technolgoits wic hekp u to derive and come up.
#import from the system as well as the system from the backend as well as we need to do.


import cv2
import dlib
import numpy as np
from scipy.spatial import distance as dist
import streamlit as st

# Initialize Streamlit app settings
st.set_page_config(layout="wide")
st.title("Real-Time Drowsiness Detection")

FRAME_WINDOW = st.image([])

# Load the face detector and landmark predictor
face_detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

# Define indices for eye landmarks
LEFT_EYE_POINTS = list(range(36, 42))
RIGHT_EYE_POINTS = list(range(42, 48))

# Drowsiness detection constants
EYE_AR_THRESHOLD = 0.25  # Threshold for eye aspect ratio
CONSEC_FRAMES = 20       # Number of frames for which eye should remain closed

# Helper function to compute Eye Aspect Ratio (EAR)
def eye_aspect_ratio(eye):
    A = dist.euclidean(eye[1], eye[5])
    B = dist.euclidean(eye[2], eye[4])
    C = dist.euclidean(eye[0], eye[3])
    ear = (A + B) / (2.0 * C)
    return ear

# Initialize counters
counter = 0
drowsy = False

# Start video capture
cap = cv2.VideoCapture(0)

while True:
    success, frame = cap.read()
    frame = cv2.flip(frame, 1)  # Flip for a mirror effect
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces
    faces = face_detector(gray)

    for face in faces:
        landmarks = predictor(gray, face)

        # Get coordinates for each eye
        left_eye = np.array([(landmarks.part(i).x, landmarks.part(i).y) for i in LEFT_EYE_POINTS])
        right_eye = np.array([(landmarks.part(i).x, landmarks.part(i).y) for i in RIGHT_EYE_POINTS])

        # Compute the EAR for each eye
        left_ear = eye_aspect_ratio(left_eye)
        right_ear = eye_aspect_ratio(right_eye)
        ear = (left_ear + right_ear) / 2.0

        # Check if EAR is below threshold (indicating closed eyes)
        if ear < EYE_AR_THRESHOLD:
            counter += 1
            if counter >= CONSEC_FRAMES:
                drowsy = True
                st.subheader("Drowsy")
        else:
            drowsy = False
            counter = 0
            st.subheader("Awake")

        # Draw green box around the detected face
        (x, y, w, h) = (face.left(), face.top(), face.width(), face.height())
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

    # Display the frame with a green box around face
    FRAME_WINDOW.image(frame, channels="BGR")

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
# Code auto enerated!!
cap.release()
cv2.destroyAllWindows()
