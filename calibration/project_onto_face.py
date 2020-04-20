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
# to inport method in base folder
import sys
sys.path.insert(1, '../')
sys.path.insert(1, '../TrainedNN')
import face_methods
from face_classifier import FaceClassifier
import copy

import os

os.environ['KMP_DUPLICATE_LIB_OK']='True'

project_square = False

# todo: consider only looking at the points around the face to speed up face detection for points

# gets the mapping from projected checkerboard to camera received checkerboard
# Can then use the mapping to linearly interpolate
def get_checkerboard_mapping():
	checker_rows = 9
	checker_cols = 6
	objp = np.zeros((checker_rows*checker_cols,3), np.float32)
	objp[:,:2] = np.mgrid[0:checker_cols,0:checker_rows].T.reshape(-1,2)

	# produce checkerboard to put on screen
	cv2.namedWindow("checkerboard")
	checkerboard_image = io.imread("checkerboard.png")
	cv2.imshow("checkerboard", checkerboard_image)

	cv2.namedWindow("result")

	while True:
		print("got here")
		cam = cv2.VideoCapture(0)
		ret, frame = cam.read()
		gray_frame = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)

		ret1, corners = cv2.findChessboardCorners(gray_frame, (checker_rows,checker_cols),None)

		frame = cv2.drawChessboardCorners(frame, (checker_rows,checker_cols), corners,ret1)

		cv2.imshow("result", frame)

		k = cv2.waitKey(0)
		if (k%256 == 27):
			print("Escape hit, closing...")
			print("camera coordinates")
			print(corners)
			print("projector coordinates")
			print(objp)
			break

# give it a coordinate in camera frame and it will return where it should be in the projector frame
def get_camera_position(camera_coordinates, projector_coordinates, camera_position_x, camera_position_y):
	# find nearest points in camera_coordinates
	#for 
	# find the slope needed
	#for
	pass 

# basically a helper method to make it easier to switch in and out techniques to get from camera to projector
def find_camera_to_projection(camera_pos):
	(camera_pos_x, camera_pos_y) = (camera_pos[0], camera_pos[1])
	print("camera pos x: %f y: %f" %(camera_pos_x, camera_pos_y)) 
	(x_origin, y_origin) = (831, 240)
	(x_proj, y_proj) = (390, 200)
	dx = 30.0/50 * 3.3
	dy = 30.0/60 * 3.5
	return np.array([x_proj+(camera_pos_x - x_origin)*dx,y_proj+(camera_pos_y- y_origin)*dy])



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
    #image = imutils.resize(image, width=500)

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
            if(j == 29):
            	return np.array([mark[0],mark[1]])
    return None

def get_face_coordinates(predictor_model, image):
	# Get detectors and predictors
    face_detector = dlib.get_frontal_face_detector()
    landmark_predictor = dlib.shape_predictor(predictor_model)

    # Load the image
    #image = imutils.resize(image, width=500)

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
        top = 255
        bottom = 0
        left = 500
        right = 0
        
        j = 0
        # eye landmarks
        # 36-41 42-47
        for mark in landmarks_arr:
#            print type(mark)
            if(top > mark[1]):
                top = mark[1]
            if(bottom < mark[1]):
                bottom = mark[1]
            if(left >mark[0]):
                left = mark[0]
            if(right < mark[0]):
                right = mark[0]
        return np.array([left, top, right, bottom])
    return np.array([])


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

	cv2.namedWindow("face")
	# create the Tkinter sceen
	"""
	win = GraphWin('screen', 1500, 1000)
	rect = Rectangle(Point(20, 10), Point(40,40))
	rect.setFill('blue')
	rect.draw(win)
	"""


	scene_to_image_matrix = np.linalg.inv(image_to_scene_matrix)

	# allow time to put the screen in right area
	cv2.waitKey(0)

	print("creating face classifier")
	#fc = FaceClassifier()

	while True:
		ret, frame = cam.read()
		#get_black_eye_coordinates
		print("started classification")
		#warped_face = face_methods.worp(frame, predictor_model)
		#cv2.imshow("face", warped_face)
		print("predict")
		#predicted_face = fc.predict_bounded_face(warped_face)
		print("classification")
		#classification = fc.identify_pred(predicted_face)

		print("getting landmarks")
		corners = get_face_coordinates(predictor_model, frame)

		# image modification
		#cv2.putText(frame, classification, (200, 500), cv2.FONT_HERSHEY_SIMPLEX, 5.0, (255,100,100), thickness=10)
		#cv2.imshow("cam_view", frame)

		cv2.waitKey(0)

		checker_rows = 9
		checker_cols = 6
		
		if(project_square):
			continue
			if(corners != None):
				#print(corners)
				#print(corners[1,0,1].size)
				project_point = np.array([corners[0],corners[1], 0])
				project_point = find_camera_to_projection(project_point)
				#project_point = project_point.dot(scene_to_image_matrix)
				print(project_point)
				"""
				rectdelete = Rectangle(Point(0, 0), Point(width_screen,height_screen))
				rectdelete.setFill('black')
				rectdelete.draw(win)
				rect = Rectangle(Point(project_point[0], project_point[1]), Point(project_point[0]+square_size, project_point[1]+square_size))
				rect.setFill('blue')
				rect.draw(win)
				"""
		else:
			print(corners)
			if(corners.size != 0):
				projector_point1 = np.array([corners[0], corners[1], 0])
				projector_point2 = np.array([corners[2], corners[3], 0])
				projector_point1 = find_camera_to_projection(projector_point1)
				projector_point2 = find_camera_to_projection(projector_point2)
				white_background = io.imread("white_background.png")
				print(white_background.shape)
				white_background = white_background[:,:,1:4]
				print(white_background.shape)
				face = io.imread("../invertwarp.jpg")
				width = int(projector_point2[0] - projector_point1[0])
				height = int(projector_point2[1] - projector_point1[1])
				new_size = (width,height)
				resized_face = cv2.resize(face, new_size)
				print(resized_face.shape)
				print(int(projector_point1[0]))
				print(width)
				white_background[int(projector_point1[0]):int(projector_point1[0])+width, int(projector_point1[1]):int(projector_point1[1])+height] = resized_face
				#face.copyTo(white_background(cv2.Rect(projector_point1[0], projector_point1[1], width, (height))))
				cv2.imshow("face", white_background)


		#canvas.create_rectangle(x0,y0,x1,y1, fill=color)


def main():
    predictor_model = sys.argv[1]

    #get_checkerboard_mapping()
    project(predictor_model)
    #dlib.hit_enter_to_continue()


if __name__ == "__main__":
    main()