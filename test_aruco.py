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
from shapely.geometry import Polygon

hit_box_scale = 1.2
hit_box_scale_grip = 2.5

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
    #print(corners-center)
    ret = np.array([(np.dot(scale_matrix,((corners-center)[0]).T) + center.T).T], dtype=np.float32)
    #print(ret)
    return ret


def robot_move(tvec, height, rot):
    x, y, z, roll, pitch, yaw = robot.get_position()
    wps = [
        [x, y, z, roll, pitch, yaw],
        [[tvec[0]], [tvec[1]], [height], [0], pitch, [rot]],
    ]
    try:
        print('moving', wps)
        robot.execute_cart_trajectory(wps)
        print('done')
    except:
        print('failed')

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
    #print(corners)
    #print(ids)
    #print('-----------')

    # Estimate SE3 pose of the marker
    camera_matrix = calibration_dict['K']

    distortion = calibration_dict['distortion']
    tvec_robot = None
    cube_rotation = None

    gripable = []
    
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
            #mtvec[0,0,1] *= 0.912

            tvec_robot[0] = -tvec_robot[0] + (0.149 + 0.324)
            tvec_robot[1] = tvec_robot[1] + (0.635 - 0.228)

            R_ct    = np.matrix(cv2.Rodrigues(rvec)[0])
            R_tc    = R_ct.T
            roll_marker, pitch_marker, cube_rotation = rotationMatrixToEulerAngles(R_flip*R_tc)

            scales = [np.array([hit_box_scale_grip, hit_box_scale]), np.array([hit_box_scale, hit_box_scale_grip])]
            for orientation, scale in enumerate(scales):
                corners_centered = scaleHitBoxes(corners[i], scale, cube_rotation)
                cv2.aruco.drawDetectedMarkers(frame, [corners_centered], np.array([ids[i]]))

                if len(ids) == 1:
                    gripable.append((tvec_robot.copy(), cube_rotation, orientation))
                    break

                hit_box = Polygon(corners_centered[0])

                for j in range(len(ids)):
                    if j == i:
                        continue

                    other_hit_box = Polygon(corners[j][0])
                    intersects = hit_box.intersects(other_hit_box)

                    if intersects:
                        cv2.fillPoly(frame, [corners_centered.astype(np.int32)], color=(0, 0, 255))
                        break
                else:
                    gripable.append((tvec_robot.copy(), cube_rotation, orientation))


            #print(tvec_robot, rvec, cube_rotation)

    #print('gripable', gripable)

    cv2.imshow("Image with frames", frame)
    key = cv2.waitKey(1)

    if key == ord('q'):
        exit()
    
    elif key == ord('m') and gripable:
        tvec_robot, cube_rotation, orient = gripable[0]
        cube_rotation += np.pi/2 * orient

        coast_height = 0.150
        grip_height = coast_height-0.06
        cube_rotation = np.pi/2 - np.mod(cube_rotation + np.pi/2, np.pi)

        robot_move(tvec_robot, coast_height, cube_rotation)
        robot_move(tvec_robot, grip_height, cube_rotation)

        robot.set_gripper('close')

        robot_move(tvec_robot, coast_height, cube_rotation)
        robot_move([0.378, 0.641], coast_height, 0)

        robot.set_gripper('open')

