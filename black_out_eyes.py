import sys
import dlib
import imutils
from imutils import face_utils
import cv2
import numpy as np
from skimage import io

"""
    Example usage: python face_landmarks.py
                   shape_predictor_68_landmarks.dat image.jpg
"""


def get_landmarks(predictor_model, image_file):
    """ Identifies face landmarks and draws them on the screen

        Args:
            predictor_model (str): Path to the pre-trained model
            image_file (str): Path to the image file
    """

    # Get detectors and predictors
    face_detector = dlib.get_frontal_face_detector()
    landmark_predictor = dlib.shape_predictor(predictor_model)
    win = dlib.image_window()

    # Load the image
    image = io.imread(image_file)
    image = imutils.resize(image, width=500)

    # Run the HOG face detector on the image data
    detected_faces = face_detector(image, 1)

    # Show the desktop window with the image
    win.set_image(image)

    # Loop through the faces found
    for i, face_rect in enumerate(detected_faces):
        # Draw a box around each face
        win.add_overlay(face_rect)
        # Get the landmarks
        landmarks = landmark_predictor(image, face_rect)
        # Convert landmarks to numpy array
        landmarks_arr = face_utils.shape_to_np(landmarks)
        # print(landmarks_arr)

        # Draw the landmarks on the screen
        win.add_overlay(landmarks)
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
        cv2.rectangle(image, (left_eye,top_eye), (right_eye, bottom_eye), (0,0,0,1.0), -1)
        cv2.imwrite("eye rectangle.jpg", image)


def main():
    predictor_model = sys.argv[1]
    image_file = sys.argv[2]

    get_landmarks(predictor_model, image_file)
    dlib.hit_enter_to_continue()


if __name__ == "__main__":
    main()

