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
import copy

np.set_printoptions(formatter={'float': lambda x: "{0:0.3f}".format(x)})

hit_box_scale = 1.2
hit_box_scale_grip = 2.5

R_flip  = np.zeros((3,3), dtype=np.float32)
R_flip[0,0] = 1.0
R_flip[1,1] =-1.0
R_flip[2,2] =-1.0

calib_z1 = 0.099
calib_x1 = -0.016
calib_y1 = 0.492
calib_z2 = 0.070
calib_x2 = 0.367
calib_y2 = 0.759

new_calib_x1 = 0.015
new_calib_y1 = 0.761
new_calib_z1 = 0.324
new_calib_x2 = 0 
new_calib_y2 = 0.756
new_calib_z2 = 0.011

TARGET_IDS = [11, 21, 22, 23, 18, 4]
TARGET_MAPPER = {2: 11, 3: 21}
ARUCO_SIZE = 0.038
TARGET_RADIUS = ARUCO_SIZE * 3

levels = [0.01, 0.07, 0.128, 0.134]

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
        [[tvec[0]], [tvec[1]], [height + 0.027], [0], pitch, [rot]],
    ]
    try:
        print('moving', wps)
        robot.execute_cart_trajectory(wps)
        print('done')
    except:
        print('failed')


def get_robot_height(camera_height):
    STEP_CAMERA = 0.05
    STEP_ROBOT = 0.136-0.086 
    OFFSET_CAMERA = STEP_CAMERA/2.5
    OFFSET_ROBOT = STEP_ROBOT/3 + 0.034
    
    level = int((camera_height + OFFSET_CAMERA)/STEP_CAMERA)
    level = max(1, level)
    return level * STEP_ROBOT + OFFSET_ROBOT

def get_coords(corner):
    rvec, tvec = cv2.aruco.estimatePoseSingleMarkers(
        corner, ARUCO_SIZE, camera_matrix, distCoeffs=distortion
    )
    tvec_robot = np.zeros(3)
    tvec_robot[0] = -tvec[0,0,0] + 0.157
    tvec_robot[1] = tvec[0,0,1] + 0.632
    tvec_robot[2] = -tvec[0,0,2] + (1.61 + 0.013)

    tvec_robot[2] += ((tvec_robot[0]-calib_x1)/(calib_x2-calib_x1)*0.5 + \
            (tvec_robot[1]-calib_y1)/(calib_y2-calib_y1)*0.5)*(calib_z1-calib_z2) - calib_z2

    print(tvec_robot, '-before')
    #tvec_robot[0] -= tvec[0,0,2]*(new_calib_x2 - new_calib_x1)/(new_calib_z2-new_calib_z1)*0.95
    #tvec_robot[1] -= tvec[0,0,2]*(new_calib_y2 - new_calib_y1)/(new_calib_z2-new_calib_z1)*3 

    print(tvec_robot, '-after')

    return tvec_robot, rvec

def calc_max_height(corners):
    max_height = -float('inf')
    for corner in corners:
        tvec_robot, rvec = get_coords(corner)
        max_height = max(tvec_robot[2], max_height)

    return get_robot_height(max_height)

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
robot.set_max_speed(0.3)

target_positions = {}
record_target_positions = False

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
    #frame = gray

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
    z_avg = 0

    if (ids is not None) and len(ids) != 0:
        max_height = calc_max_height(corners)
        for i, cube_id in enumerate(ids):
            cube_id = cube_id[0]
            tvec_robot, rvec = get_coords(corners[i])
            z_avg += tvec_robot[2]
            R_ct = np.matrix(cv2.Rodrigues(rvec)[0])
            R_tc = R_ct.T
            roll_marker, pitch_marker, cube_rotation = rotationMatrixToEulerAngles(R_flip*R_tc)

            if cube_id in TARGET_IDS:
                # target box handling
                cv2.aruco.drawDetectedMarkers(frame, [corners[i]], np.array([cube_id], dtype=np.int32))
                if record_target_positions:
                    print('recorded target', cube_id, 'at', tvec_robot)
                    target_positions[cube_id] = tvec_robot.copy()

            else:
                # all other cube handling
                if get_robot_height(tvec_robot[2]) != max_height:
                    continue

                if any([np.linalg.norm(t - tvec_robot) < TARGET_RADIUS for t in target_positions.values()]):
                    continue

                scales = [np.array([hit_box_scale_grip, hit_box_scale]), np.array([hit_box_scale, hit_box_scale_grip])]
                for orientation, scale in enumerate(scales):
                    corners_centered = scaleHitBoxes(corners[i], scale, cube_rotation)
                    cv2.aruco.drawDetectedMarkers(frame, [corners_centered], np.array([ids[i]]))

                    if len(ids) == 1:
                        gripable.append((tvec_robot.copy(), cube_rotation, orientation))
                        break

                    hit_box = Polygon(corners_centered[0])

                    for j, other_id in enumerate(ids):
                        if j == i:
                            continue

                        tvec_robot_other, rvec_other = get_coords(corners[j])
                        if get_robot_height(tvec_robot_other[2]) != max_height:
                            continue

                        other_hit_box = Polygon(corners[j][0])
                        intersects = hit_box.intersects(other_hit_box)

                        if intersects:
                            cv2.fillPoly(frame, [corners_centered.astype(np.int32)], color=(0, 0, 255))
                            break
                    else:
                        gripable.append((tvec_robot.copy(), cube_rotation, orientation, cube_id))

    #print('gripable', gripable)
    record_target_positions = False
    cv2.imshow("Image with frames", frame)
    key = cv2.waitKey(1)
    print(z_avg/4)
    
    if key == ord('q'):
        exit()
    
    elif key == ord('m') and gripable and len(gripable):
        tvec_robot, cube_rotation, orient, cube_id = gripable[0]

        cube_rotation += np.pi/2 * orient
        drop_height = 0.2

        coast_height = get_robot_height(tvec_robot[2])
        print(coast_height)
        grip_height = coast_height-0.04
        coast_height += 0.1
        cube_rotation = np.pi/2 - np.mod(cube_rotation + np.pi/2, np.pi)

        robot_move(tvec_robot, coast_height, cube_rotation)
        robot_move(tvec_robot, grip_height, cube_rotation)

        robot.set_gripper('close')

        robot_move(tvec_robot, coast_height, cube_rotation)
        
        try:
            pos = target_positions[TARGET_MAPPER[cube_id]]
            robot_move(pos, coast_height, 0)
            if coast_height > drop_height:
                robot_move(pos, drop_height, 0)
        except IndexError:
            print(f'No target for cube id={cube_id}')
            robot_move([0.378, 0.641], coast_height, 0)

        robot.set_gripper('open')

    elif key == ord('t') and gripable and len(gripable):
        tvec_robot, cube_rotation, orient = gripable[0]
        cube_rotation += np.pi/2 * orient
        coast_height = get_robot_height(tvec_robot[2])
        cube_rotation = np.pi/2 - np.mod(cube_rotation + np.pi/2, np.pi)
        robot_move(tvec_robot, coast_height, cube_rotation)

    elif key == ord('r'):
        target_positions = {}
        record_target_positions = True

        robot_move([0.565, 0.510], 0.2, 0)
        robot.set_gripper('open')

