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

# assuming 1 face and will just resize the image to 500 pixels wide
# returns all the points found on a face
def get_default_landmarks(image):
    face_detector = dlib.get_frontal_face_detector()
    landmark_predictor = dlib.shape_predictor(shape_predictor_68_landmarks.dat)

    image = imutils.resize(image, width=500)

    detected_faces = face_detector(image, 1)

    landmarks = landmark_predictor(image, detected_faces[0])
    # Convert landmarks to numpy array
    landmarks_arr = face_utils.shape_to_np(landmarks)
    return landmarks_arr

def main():
    predictor_model = sys.argv[1]
    image_file = sys.argv[2]

    get_landmarks(predictor_model, image_file)
    dlib.hit_enter_to_continue()


if __name__ == "__main__":
    main()

