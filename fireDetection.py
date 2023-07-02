
import sys
import os
import cv2
from playsound import playsound

def resource_path(relative_path):
    try:
        base_path = sys._MEIPASS # type: ignore
    except Exception:
        base_path = os.path.abspath(".")

    return os.path.join(base_path, relative_path)

xml_path = resource_path('fire_detection.xml')
fire_cascade = cv2.CascadeClassifier(xml_path)

cap = cv2.VideoCapture("video.mp4")

# contineoulsy capturing the video
while(True):
    ret, frame = cap.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    fire = fire_cascade.detectMultiScale(frame, 1.2, 5)
    # detected value = false , as no fire detected
    detected = False
    for (x,y,w,h) in fire:
        cv2.rectangle(frame,(x-20,y-20),(x+w+20,y+h+20),(255,0,0),2)
        roi_gray = gray[y:y+h, x:x+w]
        roi_color = frame[y:y+h, x:x+w]
        # printing when fire detected
        print("fire is detected")
        detected = True
        # playing the sound when the fire is detected in the frame
        playsound(resource_path('audio.mp3'))
    if detected:
        break
    cv2.imshow(':: Fire detection Model :: ', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
