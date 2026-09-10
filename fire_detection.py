import cv2
import numpy as np
import winsound
import threading


# Start the webcam
cap = cv2.VideoCapture(0)

# Keeps track of the alarm
alarm_on = False

def play_alarm():
    """Play the alarm sound."""
    winsound.PlaySound(
        "alarm.wav",
        winsound.SND_FILENAME
    )


while True:

    # Read a frame from the camera
    ret, frame = cap.read()

    if not ret:
        print("Unable to read camera frame.")
        break

    # Convert the image to HSV
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # Fire-like color range
    lower_fire = np.array([0, 120, 120])
    upper_fire = np.array([50, 255, 255])

    # Create a mask for the selected colors
    fire_mask = cv2.inRange(
        hsv,
        lower_fire,
        upper_fire
    )

    # Count the number of detected fire-colored pixels
    fire_pixels = cv2.countNonZero(fire_mask)

    # Show the mask for testing
    cv2.imshow("Fire Mask", fire_mask)

    # Check whether enough fire-colored pixels are present
    if fire_pixels > 1500:
        cv2.putText(
            frame,
            "FIRE DETECTED",
            (50, 50),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 0, 255),
            3
        )

        # Start the alarm only once
        if not alarm_on:
            alarm_on = True

            alarm_thread = threading.Thread(
                target=play_alarm
            )
            alarm_thread.start()
    else:
        alarm_on = False

    # Display the camera feed
    cv2.imshow(
        "Fire Detection Camera",
        frame
    )

    # Press Q to close the program
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

# Release camera and close all windows
cap.release()
cv2.destroyAllWindows()
