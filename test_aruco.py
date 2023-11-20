# import the necessary packages
import argparse
import numpy as np
import time
import cv2
import sys
import PyCapture2
import pickle
import rospy
from mitsubishi_arm_student_interface.mitsubishi_robots import Mitsubishi_robot


x_center = -0.16402563 - 0.025
y_center = -0.25913066 - 0.025

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
    return np.array(rvecs), np.array(tvecs), np.array(trash)



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

# Initialize robot interface class
robot = Mitsubishi_robot()

# Set maximal relative speed (it is recomended to decrease the speed for testing)
robot.set_max_speed(0.1);

with open('calibration.pickle', 'rb') as f:
    calibration_dict = pickle.load(f)

while True:
    image = camera.retrieveBuffer()

    # Convert from MONO8 to RGB8
    image = image.convert(PyCapture2.PIXEL_FORMAT.RGB8)

    # Convert image to Numpy array
    frame = np.array(image.getData(), dtype="uint8").reshape(
        (image.getRows(), image.getCols(), 3)
    )

    frame = cv2.undistort(
            frame,
            calibration_dict['K'],
            calibration_dict['distortion'],
            None,
            calibration_dict['Knew']
    )

    # RGB to BGR and grayscale
    frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # adjust
    gray = cv2.convertScaleAbs(gray, alpha=2, beta=0)

    # detect
    arucoDict = cv2.aruco.Dictionary_get(cv2.aruco.DICT_6X6_50)
    arucoParams = cv2.aruco.DetectorParameters_create()
    (corners, ids, rejected) = cv2.aruco.detectMarkers(gray, arucoDict,
	parameters=arucoParams)

    cv2.aruco.drawDetectedMarkers(gray, corners, ids)
    cv2.aruco.drawDetectedMarkers(frame, corners, ids)

    # Estimate SE3 pose of the marker
    camera_matrix = calibration_dict['K']

    distortion = calibration_dict['distortion']

    if (ids is not None) and len(ids) != 0:
        for i in range(len(ids)):
            rvec, tvec = cv2.aruco.estimatePoseSingleMarkers(
                corners[i], 0.04, camera_matrix, distCoeffs=distortion
            )
            cv2.aruco.drawAxis(frame, camera_matrix, distortion, rvec, tvec, 0.04)
            tvec = tvec - np.array([[[x_center, y_center, 0]]])

            tvec_robot = tvec
            tvec_robot[0,0,0] = -tvec_robot[0,0,0] + 0.470
            tvec_robot[0,0,1] = tvec_robot[0,0,1] + 0.415
            print(tvec_robot)


    cv2.imshow("Image with frames", frame)
    key = cv2.waitKey(1)

    if key == ord('q'):
        exit()
    
    elif key == ord('m') and (tvec_robot is not None):
        x, y, z, roll, pitch, yaw = robot.get_position()

        wps = [
            [x, y, z, roll, pitch, yaw],
            [tvec_robot[0,0], tvec_robot[0,1], z, roll, pitch, yaw],
        ]

        #robot.execute_cart_trajectory(wps)

