import numpy as np
import cv2 as cv
import glob
import pickle

CHESS_WIDTH = 12
CHESS_HEIGHT = 8

# termination criteria
criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 25, 0.001)
# prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
objp = np.zeros((CHESS_WIDTH*CHESS_HEIGHT,3), np.float32)
objp[:,:2] = np.mgrid[0:CHESS_WIDTH,0:CHESS_HEIGHT].T.reshape(-1,2)
# Arrays to store object points and image points from all the images.
objpoints = [] # 3d point in real world space
imgpoints = [] # 2d points in image plane.

images = glob.glob('calibrate/*_color.jpg')
print('found:', images)

for fname in images:
    img = cv.imread(fname)
    img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    # Find the chess board corners
    ret, corners = cv.findChessboardCorners(img, (CHESS_WIDTH, CHESS_HEIGHT), None)
    print(fname, ret)
    #cv.imshow('input', img)
    #cv.waitKey(500)
    # If found, add object points, image points (after refining them)
    if ret == True:
        objpoints.append(objp)
        corners2 = cv.cornerSubPix(img, corners, (11,11), (-1,-1), criteria)
        imgpoints.append(corners2)
        # Draw and display the corners
        #cv.drawChessboardCorners(img, (12, 8), corners2, ret)
        #cv.imshow('img', img)
        #cv.waitKey(1000)

_, K, distortion, rvecs, tvecs = cv.calibrateCamera(objpoints, imgpoints, img.shape[::-1], None, None)
Knew, _ = cv.getOptimalNewCameraMatrix(K, distortion, img.shape[::-1], 1, img.shape[::-1])

with open('calibration.pickle', 'wb') as f:
    pickle.dump({
        'K' : np.array(K),
        'distortion' : distortion,
        'Knew' : np.array(Knew)
    }, f, protocol=1)

cv.destroyAllWindows()

