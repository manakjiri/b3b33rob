# import the necessary packages
import argparse
import numpy as np
import time
import cv2
import sys
import PyCapture2

# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-t", "--type", type=str,
	default="DICT_6X6_50",
	help="type of ArUCo tag to detect")
args = vars(ap.parse_args())

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
    frame = np.array(image.getData(), dtype="uint8").reshape((image.getRows(), image.getCols(), 3));
     
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_6X6_50)
    detector = cv2.aruco.ArucoDetector(aruco_dict)

    # Detect markers and draw them
    (corners, ids, rejected) = detector.detectMarkers(gray)
    cv2.aruco.drawDetectedMarkers(frame, corners, ids)

    cv2.imshow("Image with markers", frame)
    cv2.waitKey()

    # Estimate SE3 pose of the marker
    camera_matrix = np.array(
        [
            [240.0, 0, 0],
            [0, 240, 0],
            [0, 0, 1],
        ]
    )
    distortion = np.zeros(5)
    for i in range(len(ids)):
        rvec, tvec, _ = cv2.aruco.estimatePoseSingleMarkers(
            corners[i], 0.04, camera_matrix, distCoeffs=distortion
        )
        cv2.drawFrameAxes(frame, camera_matrix, distortion, rvec, tvec, 0.04)

        print(tvec)

    cv2.imshow("Image with frames", frame)
    cv2.waitKey()
