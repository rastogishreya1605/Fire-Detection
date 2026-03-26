import cv2
import numpy as np
import winsound
import threading

cap = cv2.VideoCapture(0)

alarm_on = False   # alarm status

def play_alarm():
    winsound.PlaySound("alarm.wav", winsound.SND_FILENAME)

while True:

    ret, frame = cap.read()
    if not ret:
        break

    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    lower_fire = np.array([0,120,120])
    upper_fire = np.array([50,255,255])

    mask = cv2.inRange(hsv, lower_fire, upper_fire)

    fire_pixels = cv2.countNonZero(mask)

    cv2.imshow("Fire Mask", mask)

    if fire_pixels > 1500:

        cv2.putText(frame,"FIRE DETECTED",(50,50),
                    cv2.FONT_HERSHEY_SIMPLEX,1,(0,0,255),3)

        if not alarm_on:
            alarm_on = True
            threading.Thread(target=play_alarm).start()

    else:
        alarm_on = False

    cv2.imshow("Fire Detection Camera", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()