import sys
import dlib
import imutils
from imutils import face_utils
import cv2
import numpy as np
from skimage import io
import math

"""
    Example usage: python face_landmarks.py
                   shape_predictor_68_landmarks.dat image.jpg
"""

# euclidean distance
def distance(point_1, point_2):
    dist = pow(point_1[0]-point_2[0],2) + pow(point_1[1]-point_2[1],2)
    return pow(dist,.5)


def get_landmarks(predictor_model, image_file_1, image_file_2):
    """ Identifies face landmarks and draws them on the screen

        Args:
            predictor_model (str): Path to the pre-trained model
            image_file (str): Path to the image file
    """

    # Get detectors and predictors
    face_detector = dlib.get_frontal_face_detector()
    landmark_predictor = dlib.shape_predictor(predictor_model)

    # Load the image
    image_1 = io.imread(image_file_1)
    image_1 = imutils.resize(image_1, width=500)

    image_2 = io.imread(image_file_2)
    image_2 = imutils.resize(image_2, width=500)

    # Run the HOG face detector on the image data
    detected_face_1 = face_detector(image_1, 1)
    detected_face_2 = face_detector(image_2, 1)

    face_1 = detected_face_1[0]
    face_2 = detected_face_2[0]

    landmarks_1 = landmark_predictor(image_1, face_1)
    landmarks_2 = landmark_predictor(image_2, face_2)

    landmarks_arr_1 = face_utils.shape_to_np(landmarks_1)
    landmarks_arr_2 = face_utils.shape_to_np(landmarks_2)

    if(len(landmarks_arr_1) != len(landmarks_arr_2)):
        return
    for i in range(len(landmarks_arr_1)):
        cv2.rectangle(image_1, (landmarks_arr_1[i][0],landmarks_arr_1[i][1]), (landmarks_arr_1[i][0]+10,landmarks_arr_1[i][1]+10), (4*(distance(landmarks_arr_1[i], landmarks_arr_2[i])-110),0,0,1.0), -1)
    cv2.imwrite("distance.jpg", image_1)


def main():
    predictor_model = sys.argv[1]
    image_file_1 = sys.argv[2]
    image_file_2 = sys.argv[3]

    get_landmarks(predictor_model, image_file_1, image_file_2)


if __name__ == "__main__":
    main()

