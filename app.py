import streamlit as st
import cv2
import numpy as np
import av

from streamlit_webrtc import webrtc_streamer, VideoProcessorBase


# ---------------------------------------------------------
# App settings
# ---------------------------------------------------------

st.set_page_config(
    page_title="Fire Detection AI",
    page_icon="🔥",
    layout="wide"
)


# ---------------------------------------------------------
# App heading
# ---------------------------------------------------------

st.title("🔥 Fire Detection AI")

st.write(
    "This application uses computer vision to detect "
    "possible fire in real time."
)

st.info(
    "Click the START button below and allow camera access "
    "when your browser asks for permission."
)


# ---------------------------------------------------------
# Fire detection
# ---------------------------------------------------------

class FireDetector(VideoProcessorBase):

    def recv(self, frame):

        # Get the camera image
        image = frame.to_ndarray(format="bgr24")

        # Make the image smaller so detection is faster
        image = cv2.resize(image, (960, 540))

        # Convert image into HSV format
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

        # These values describe common fire colors
        lower_fire = np.array([0, 120, 120])
        upper_fire = np.array([50, 255, 255])

        # Find pixels that look like fire
        fire_mask = cv2.inRange(
            hsv,
            lower_fire,
            upper_fire
        )

        # Remove small noise
        fire_mask = cv2.medianBlur(
            fire_mask,
            9
        )

        # Count possible fire pixels
        fire_pixels = cv2.countNonZero(
            fire_mask
        )

        # -------------------------------------------------
        # Show result
        # -------------------------------------------------

        if fire_pixels > 1500:

            cv2.rectangle(
                image,
                (0, 0),
                (959, 539),
                (0, 0, 255),
                15
            )

            cv2.putText(
                image,
                "FIRE DETECTED!",
                (40, 80),
                cv2.FONT_HERSHEY_SIMPLEX,
                2,
                (0, 0, 255),
                5
            )

        else:

            cv2.putText(
                image,
                "NO FIRE DETECTED",
                (40, 80),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.4,
                (0, 255, 0),
                4
            )

        # Show number of detected fire pixels
        cv2.putText(
            image,
            f"Fire pixels: {fire_pixels}",
            (40, 125),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (255, 255, 255),
            2
        )

        # Send the processed camera image back
        return av.VideoFrame.from_ndarray(
            image,
            format="bgr24"
        )


# ---------------------------------------------------------
# Start camera
# ---------------------------------------------------------

webrtc_streamer(
    key="fire-camera",
    video_processor_factory=FireDetector,
    media_stream_constraints={
        "video": True,
        "audio": False
    },
    async_processing=True
)


# ---------------------------------------------------------
# Small note
# ---------------------------------------------------------

st.caption(
    "Note: This is a computer-vision based demonstration "
    "and should not be used as a professional fire alarm system."
)