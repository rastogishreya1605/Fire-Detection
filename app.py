import streamlit as st
import cv2
import numpy as np
import time
import os
try:
    import pygame
    pygame_available = True
except ImportError:
    pygame_available = False

# 1. Sound File Setup
ALARM_FILE = "alarm.wav" 

# Pygame Mixer Initialize
pygame.mixer.init()

# 2. Page Config (Full Screen & Clean)
st.set_page_config(page_title="Fire Detection AI", layout="wide", initial_sidebar_state="collapsed")

st.markdown("""
    <style>
    [data-testid="stHeader"], [data-testid="stFooter"], [data-testid="stSidebar"] {display: none;}
    .block-container {padding: 0rem;}
    </style>
    """, unsafe_allow_html=True)

FRAME_WINDOW = st.image([])

# 3. Main Logic
camera = cv2.VideoCapture(0)

while True:
    ret, frame = camera.read()
    if not ret:
        break

    # Full screen resize
    frame = cv2.resize(frame, (1280, 720)) 
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    
    # Fire Detection (Color Range)
    lower_fire = np.array([0, 160, 180]) 
    upper_fire = np.array([35, 255, 255])
    
    mask = cv2.inRange(hsv, lower_fire, upper_fire)
    mask = cv2.medianBlur(mask, 9)
    fire_pixels = cv2.countNonZero(mask)

    if fire_pixels > 8000:
        # 1. Visual Alert
        cv2.rectangle(frame_rgb, (0,0), (1280, 720), (255, 0, 0), 40)
        cv2.putText(frame_rgb, "!!! FIRE DETECTED !!!", (100, 400), 
                    cv2.FONT_HERSHEY_SIMPLEX, 4, (255, 0, 0), 15)
        
        # 2. Play Alarm (Sirf tab jab pehle se na baj raha ho)
        if not pygame.mixer.music.get_busy():
            try:
                pygame.mixer.music.load(ALARM_FILE)
                pygame.mixer.music.play(-1) # -1 ka matlab loop mein bajta rahega jab tak aag hai
            except:
                pass
    else:
        # 3. FIX: Jaise hi fire hati, sound STOP!
        pygame.mixer.music.stop()
    
    # Show Camera
    FRAME_WINDOW.image(frame_rgb, width='stretch')
    
    time.sleep(0.01)

camera.release()