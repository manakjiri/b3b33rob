# import the necessary packages
import argparse
import numpy as np
import time
import cv2
import sys
import PyCapture2


def my_estimatePoseSingleMarkers(corners, marker_size, mtx, distortion):
    """
    This will estimate the rvec and tvec for each of the marker corners detected by:
       corners, ids, rejectedImgPoints = detector.detectMarkers(image)
    corners - is an array of detected corners for each detected marker in the image
    marker_size - is the size of the detected markers
    mtx - is the camera matrix
    distortion - is the camera distortion matrix
    RETURN list of rvecs, tvecs, and trash (so that it corresponds to the old estimatePoseSingleMarkers())
    """
    marker_points = np.array(
        [
            [-marker_size / 2, marker_size / 2, 0],
            [marker_size / 2, marker_size / 2, 0],
            [marker_size / 2, -marker_size / 2, 0],
            [-marker_size / 2, -marker_size / 2, 0],
        ],
        dtype=np.float32,
    )
    trash = []
    rvecs = []
    tvecs = []
    for c in corners:
        nada, R, t = cv2.solvePnP(
            marker_points, c, mtx, distortion, False, cv2.SOLVEPNP_IPPE_SQUARE
        )
        rvecs.append(R)
        tvecs.append(t)
        trash.append(nada)
    return rvecs, tvecs, trash


# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument(
    "-t", "--type", type=str, default="DICT_6X6_50", help="type of ArUCo tag to detect"
)
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
    frame = np.array(image.getData(), dtype="uint8").reshape(
        (image.getRows(), image.getCols(), 3)
    )

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
