import sys
import dlib
import imutils
from imutils import face_utils
import cv2
import numpy as np
from skimage import io

cam = cv2.VideoCapture(0)

cv2.namedWindow("test")
cv2.namedWindow("checkerboard")
checkerboard_image = io.imread("checkerboard.png")
cv2.imshow("checkerboard", checkerboard_image)

# store the points needed for image calibration
# taken from https://opencv-python-tutroals.readthedocs.io/en/latest/py_tutorials/py_calib3d/py_calibration/py_calibration.html


checker_rows = 9
checker_cols = 6
objp = np.zeros((checker_rows*checker_cols,3), np.float32)
objp[:,:2] = np.mgrid[0:checker_cols,0:checker_rows].T.reshape(-1,2)

print(objp)

objp = objp * 30
# = objp + (5,5,0)


objpoints = [] # 3d point in real world space
imgpoints = [] # 2d points in image plane.

img_counter = 0

while True:
    ret, frame = cam.read()
    gray_frame = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    
    # find chess board
    ret1, corners = cv2.findChessboardCorners(gray_frame, (checker_rows,checker_cols),None)
    # If found, add object points, image points (after refining them)
    if ret1 == True:
        objpoints.append(objp)

        #corners2 = cv2.cornerSubPix(gray,corners,(11,11),(-1,-1),criteria)
        imgpoints.append(corners)

        # Draw and display the corners
        frame = cv2.drawChessboardCorners(frame, (checker_rows,checker_cols), corners,ret1)

        M, mask = cv2.findHomography(objp, corners, cv2.RANSAC)
        print("mask")
        print(M)
    else:
        print("no good")

    cv2.imshow("test", frame)
    if not ret:
        break
    k = cv2.waitKey(0)

    if k%256 == 27:
        # ESC pressed
        #solve the matrix before leaving
        """
        ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray_frame.shape[::-1],None,None)
        h,  w = frame.shape[:2]
        newcameramtx, roi=cv2.getOptimalNewCameraMatrix(mtx,dist,(w,h),1,(w,h))
        # undistort
        dst = cv2.undistort(frame, mtx, dist, None, newcameramtx)
        x,y,w,h = roi
        dst = dst[y:y+h, x:x+w]
        cv2.imwrite('calibresult.png',dst)
        print(mtx)
        print(dist)
        print(newcameramtx)
        """

        print("Escape hit, closing...")
        break


cam.release()

cv2.destroyAllWindows()

