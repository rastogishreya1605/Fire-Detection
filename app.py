import streamlit as st
import cv2
import numpy as np
import pygame
import time
import os

# Alarm sound
ALARM_FILE = "alarm.wav"

# Start pygame audio
pygame.mixer.init()

# Streamlit page settings
st.set_page_config(
    page_title="Fire Detection AI",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Remove Streamlit default spacing and sidebar
st.markdown(
    """
    <style>
        [data-testid="stHeader"],
        [data-testid="stFooter"],
        [data-testid="stSidebar"] {
            display: none;
        }

        .block-container {
            padding: 0;
        }
    </style>
    """,
    unsafe_allow_html=True
)

# Camera display
camera_view = st.image([])


# Open webcam
camera = cv2.VideoCapture(0)

while True:
    success, frame = camera.read()

    # Camera frame ko screen ke according resize karna
    frame = cv2.resize(frame, (1280, 720))

    # OpenCV BGR mein image deta hai, Streamlit ke liye RGB chahiye
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # HSV image fire color identify karne mein useful hai
    hsv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # Fire ke liye approximate color range
    lower_fire = np.array([0, 160, 180])
    upper_fire = np.array([35, 255, 255])
    fire_mask = cv2.inRange(
        hsv_frame,
        lower_fire,
        upper_fire
    )

    # Noise reduce
    fire_mask = cv2.medianBlur(fire_mask, 9)
    fire_pixels = cv2.countNonZero(fire_mask)
    if fire_pixels > 8000:

        # Screen par red border
        cv2.rectangle(
            rgb_frame,
            (0, 0),
            (1279, 719),
            (255, 0, 0),
            40
        )

        # Warning message
        cv2.putText(
            rgb_frame,
            "FIRE DETECTED!",
            (100, 400),
            cv2.FONT_HERSHEY_SIMPLEX,
            4,(255, 0, 0), 12
        )

        # Alarm start
        if not pygame.mixer.music.get_busy():
            if os.path.exists(ALARM_FILE):
                pygame.mixer.music.load(ALARM_FILE)
                pygame.mixer.music.play(-1)

    else:
        # Fire detect na hone par alarm stop
        if pygame.mixer.music.get_busy():
            pygame.mixer.music.stop()

    # Camera frame display
    camera_view.image(
        rgb_frame,
        width="stretch"
    )
    time.sleep(0.01)

# Camera close
camera.release()
pygame.mixer.quit()

