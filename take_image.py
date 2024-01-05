import argparse
import numpy as np
import time
import cv2
import sys
import time

# Initialize bus and camera
camera = cv2.VideoCapture(0)

while True:
    _, frame = camera.read()

    # RGB to BGR and grayscale
    #frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # adjust
    gray = cv2.convertScaleAbs(gray, alpha=2, beta=0)

    cv2.imshow("Image with frames", frame)
    key = cv2.waitKey(1)

    if key == ord('q'):
        exit()
    elif key == ord('t'):
        t = int(time.time())
        cv2.imwrite(f'{t}_color.jpg', frame)
        cv2.imwrite(f'{t}_bw.jpg', gray)



