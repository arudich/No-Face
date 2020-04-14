"""
take the matrix from calibration
take a point from the image and do the inverse transform and see if the 

"""

import dlib
import imutils
from imutils import face_utils
import cv2
import numpy as np
from skimage import io
from graphics import *
# to inport method in base folder
import sys
sys.path.insert(1, '../')
import face_methods


def get_landmarks(predictor_model, image):
    """ Identifies face landmarks and draws them on the screen

        Args:
            predictor_model (str): Path to the pre-trained model
            image_file (str): Path to the image file
    """

    # Get detectors and predictors
    face_detector = dlib.get_frontal_face_detector()
    landmark_predictor = dlib.shape_predictor(predictor_model)

    # Load the image
    image = imutils.resize(image, width=500)

    # Run the HOG face detector on the image data
    detected_faces = face_detector(image, 1)

    # Loop through the faces found
    for i, face_rect in enumerate(detected_faces):
        # Draw a box around each face
        # Get the landmarks
        landmarks = landmark_predictor(image, face_rect)
        # Convert landmarks to numpy array
        landmarks_arr = face_utils.shape_to_np(landmarks)
        # print(landmarks_arr)

        # Draw the landmarks on the screen
        top_eye = 255
        bottom_eye = 0
        left_eye = 500
        right_eye = 0
        
        j = 0
        # eye landmarks
        # 36-41 42-47
        for mark in landmarks_arr:
#            print type(mark)
            if(j>35 and j<48):
                if(top_eye > mark[1]):
                    top_eye = mark[1]
                if(bottom_eye < mark[1]):
                    bottom_eye = mark[1]
                if(left_eye >mark[0]):
                    left_eye = mark[0]
                if(right_eye < mark[0]):
                    right_eye = mark[0]
            j += 1
            if(j == 25):
            	return np.array([mark[0],mark[1]])
    return None

def project(predictor_model):

	image_to_scene_matrix = np.array([[  1.86477349e+00,  -1.57771604e-03,   7.40056394e+02],
 									  [  4.17290600e-02,   1.30318147e+00,   2.27385320e+02],
 									  [  6.88206471e-05,  -6.04011568e-07,   1.00000000e+00]])


	width_screen = 1500
	height_screen = 1000
	square_size = 100


	cam = cv2.VideoCapture(0)

	# delete after testing
	cv2.namedWindow("cam_view")
	checkerboard_image = io.imread("checkerboard.png")
	cv2.imshow("cam_view", checkerboard_image)

	# create the Tkinter sceen
	win = GraphWin('screen', 1500, 1000)
	rect = Rectangle(Point(20, 10), Point(40,40))
	rect.setFill('blue')
	rect.draw(win)

	scene_to_image_matrix = np.linalg.inv(image_to_scene_matrix)

	while True:
		ret, frame = cam.read()
		#get_black_eye_coordinates

		checker_rows = 9
		checker_cols = 6
		
		corners = get_landmarks(predictor_model, frame)

		if(corners != None):
			#print(corners[1,:,:])
			#print(corners[1,0,1].size)
			project_point = np.array([corners[0],corners[1], 0])
			project_point = project_point.dot(scene_to_image_matrix)
			print(project_point)
			rectdelete = Rectangle(Point(0, 0), Point(width_screen,height_screen))
			rectdelete.setFill('white')
			rectdelete.draw(win)
			rect = Rectangle(Point(project_point[0]+500, project_point[1]), Point(project_point[0]+500+square_size, project_point[1]+square_size))
			rect.setFill('blue')
			rect.draw(win)

		#canvas.create_rectangle(x0,y0,x1,y1, fill=color)
		ret, frame = cam.read()
		cv2.imshow("cam_view", frame)
		k = cv2.waitKey(0)

		if (k%256 == 27):
			break


def main():
    predictor_model = sys.argv[1]

    project(predictor_model)
    dlib.hit_enter_to_continue()


if __name__ == "__main__":
    main()