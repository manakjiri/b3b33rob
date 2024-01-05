from scipy.spatial.transform import Rotation as R
import numpy as np
import time
import cv2
import sys
import time

CHESS_WIDTH = 9
CHESS_HEIGHT = 6
FRAME_WIDTH = 1280
FRAME_HEIGHT = 720

win_name = "Frames"

# termination criteria
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 25, 0.001)

# Initialize bus and camera
camera = cv2.VideoCapture(0)
camera.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
camera.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)

last_detect = 0
detected = False
while True:
    _, frame = camera.read()

    # RGB to BGR and grayscale
    #frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # adjust
    gray = cv2.convertScaleAbs(gray, alpha=2, beta=0)

    if time.time() - last_detect > 1 or detected:
        detected, corners = cv2.findChessboardCorners(gray, (CHESS_WIDTH, CHESS_HEIGHT), None)
        last_detect = time.time()
        if detected:
            corners2 = cv2.cornerSubPix(gray, corners, (11,11), (-1,-1), criteria)
            #cv2.drawChessboardCorners(frame, (12, 8), corners2, detected)
            points = np.float32([corners2[0][0], corners2[8][0], corners2[-1][0], corners2[-9][0]])
            print(points)
            for val in points:
                cv2.circle(frame, (int(val[0]), int(val[1])), 5, (0, 255, 0), -1)
            
            center = np.mean(points, axis=0)
            angle = np.arctan2(points[0][1] - points[1][1], points[0][0] - points[1][0])
            rotation = np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])
            points = (rotation.T @ (points - center).T).T
            max_i = np.argmax(np.linalg.norm(points, axis=1))
            scale = np.float32([FRAME_WIDTH / 2 / points[max_i][0], FRAME_HEIGHT / 2 / points[max_i][1]])
            points = points * scale + np.float32([FRAME_WIDTH / 2, FRAME_HEIGHT / 2])

            points2 = np.float32([[0, 0], [FRAME_WIDTH, 0], [FRAME_WIDTH, FRAME_HEIGHT], [0, FRAME_HEIGHT]])
            M = cv2.getPerspectiveTransform(points, points2)
            frame = cv2.warpPerspective(frame, M, (FRAME_WIDTH, FRAME_HEIGHT))

    cv2.namedWindow(win_name, cv2.WINDOW_NORMAL)
    #cv2.moveWindow(win_name, 1000, 1000)
    cv2.imshow(win_name, frame)
    cv2.resizeWindow(win_name, FRAME_WIDTH*2, FRAME_HEIGHT*2)
    key = cv2.waitKey(1)

    if key == ord('q'):
        exit()


