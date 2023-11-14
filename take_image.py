import argparse
import numpy as np
import time
import cv2
import sys
import PyCapture2
import time

# Initialize bus and camera
bus = PyCapture2.BusManager()
camera = PyCapture2.Camera()

# Select first camera on the bus
camera.connect(bus.getCameraFromIndex(0))

# Start capture
camera.startCapture()

while True:
    image = camera.retrieveBuffer()

    # Convert from MONO8 to RGB8
    image = image.convert(PyCapture2.PIXEL_FORMAT.RGB8)

    # Convert image to Numpy array
    frame = np.array(image.getData(), dtype="uint8").reshape(
        (image.getRows(), image.getCols(), 3)
    )

    # RGB to BGR and grayscale
    frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
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



