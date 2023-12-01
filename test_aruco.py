# import the necessary packagesnp.matrix(cv2.Rodrigues(rvec)[0])
import argparse
import numpy as np
import time
import cv2
import sys
import PyCapture2
import pickle
import rospy
import math
from mitsubishi_arm_student_interface.mitsubishi_robots import Mitsubishi_robot

hit_box_scale = 1.2
x_center = -0.16402563 - 0.025
y_center = -0.25913066 - 0.025 - 0.005

R_flip  = np.zeros((3,3), dtype=np.float32)
R_flip[0,0] = 1.0
R_flip[1,1] =-1.0
R_flip[2,2] =-1.0

def rotationMatrixToEulerAngles(R):
    sy = math.sqrt(R[0, 0] * R[0, 0] + R[1, 0] * R[1, 0])

    singular = sy < 1e-6

    if not singular:
        x = math.atan2(R[2, 1], R[2, 2])
        y = math.atan2(-R[2, 0], sy)
        z = math.atan2(R[1, 0], R[0, 0])
    else:
        x = math.atan2(-R[1, 2], R[1, 1])
        y = math.atan2(-R[2, 0], sy)
        z = 0

    return np.array([x, y, z])

def scaleHitBoxes(corners, scale_vector, angle):
    center = np.mean(corners, axis=1)
    rot = np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]], dtype=np.float32)
    scale_matrix = np.dot(
            rot.T,
            np.dot(
                np.diag(scale_vector),
                rot
            )
    )
    print(corners-center)
    ret = np.array([(np.dot(scale_matrix,((corners-center)[0]).T) + center.T).T], dtype=np.float32)
    print(ret)
    return ret

# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument(
    "-t", "--type", type=str, default="DICT_6X6_50", help="type of ArUCo tag to detect"
)
args = vars(ap.parse_args())

# Initialize bus and camera
bus = PyCapture2.BusManager()
camera = PyCapture2.Camera()

print(bus.getNumOfCameras())
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
    print(corners)
    print(ids)
    print('-----------')

    # Estimate SE3 pose of the marker
    camera_matrix = calibration_dict['K']

    distortion = calibration_dict['distortion']
    tvec_robot = None
    cube_rotation = None
    
    if (ids is not None) and len(ids) != 0:
        for i in range(len(ids)):
            rvec, tvec = cv2.aruco.estimatePoseSingleMarkers(
                corners[i], 0.04, camera_matrix, distCoeffs=distortion
            )
            cv2.aruco.drawAxis(frame, camera_matrix, distortion, rvec, tvec, 0.04)

            tvec_robot = np.dot(camera_matrix, tvec.flatten()) / 10 / 1000 / 1.4

            #print(tvec)
            #tvec = tvec - np.array([[[x_center, y_center, 0]]])
            #tvec[0,0,0] *= 0.96
            #tvec[0,0,1] *= 0.912

            tvec_robot[0] = -tvec_robot[0] + (0.149 + 0.324)
            tvec_robot[1] = tvec_robot[1] + (0.635 - 0.228)

            R_ct    = np.matrix(cv2.Rodrigues(rvec)[0])
            R_tc    = R_ct.T
            roll_marker, pitch_marker, cube_rotation = rotationMatrixToEulerAngles(R_flip*R_tc)

            corners_centered = scaleHitBoxes(corners[i], np.array([2,1.2]), cube_rotation)
            print([corners_centered])
            print(np.array([ids[i]]))
            cv2.aruco.drawDetectedMarkers(frame, [corners_centered], np.array([ids[i]]))


            #print(tvec_robot, rvec, cube_rotation)


    cv2.imshow("Image with frames", frame)
    key = cv2.waitKey(1)

    if key == ord('q'):
        exit()
    
    elif key == ord('m') and (tvec_robot is not None) and (cube_rotation is not None):
        x, y, z, roll, pitch, yaw = robot.get_position()
        wps = [
            [x, y, z, roll, pitch, yaw],
            [[tvec_robot[0]], [tvec_robot[1]], z, roll, pitch, yaw],
        ]
        robot.execute_cart_trajectory(wps)

        wps = [
            [[tvec_robot[0]], [tvec_robot[1]], z, roll, pitch, yaw],
            [[tvec_robot[0]], [tvec_robot[1]], z, roll, pitch, [-cube_rotation]],
        ]
        try:
            robot.execute_cart_trajectory(wps)
        except:
            pass

        wps = [
            [[tvec_robot[0]], [tvec_robot[1]], z, roll, pitch, [-cube_rotation]],
            [[tvec_robot[0]], [tvec_robot[1]], [0.130-0.04], roll, pitch, [-cube_rotation]],
        ]
        try:
            robot.execute_cart_trajectory(wps)
        except:
            pass

        robot.set_gripper('close')

        wps = [
            [[tvec_robot[0]], [tvec_robot[1]], [0.130-0.04], roll, pitch, [-cube_rotation]],
            [[0.378], [0.641], [0.130+0.04], roll, pitch, [-cube_rotation]]
        ]

        try:
            robot.execute_cart_trajectory(wps)
        except:
            pass

        robot.set_gripper('open')


