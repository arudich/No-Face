import sys
import cv2
import dlib
import numpy as np
from skimage import io

import imutils
from imutils import face_utils
from imutils.face_utils import FACIAL_LANDMARKS_IDXS


"""
    Example usage: python modify.py images/claudia/claudia_10.jpg
                   images/claudia/claudia_edited_nose.jpg nose
"""


"""
FACIAL_LANDMARKS_IDXS = OrderedDict([
    ("mouth", (48, 68)),
    ("right_eyebrow", (17, 22)),
    ("left_eyebrow", (22, 27)),
    ("right_eye", (36, 42)),
    ("left_eye", (42, 48)),
    ("nose", (27, 35)),
    ("jaw", (0, 17))
])
"""


# in BGR, not RGB!!
COLORS = [(128, 84, 231)]


def get_landmarks(image):
    """ Gets array of facial landmarks
    """

    face_detector = dlib.get_frontal_face_detector()
    landmark_predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')

    # image = cv2.imread(image_file)
    # image = imutils.resize(image, width=500)

    detected_faces = face_detector(image, 1)

    landmarks = landmark_predictor(image, detected_faces[0])
    # Convert landmarks to numpy array
    landmarks_arr = face_utils.shape_to_np(landmarks)

    return landmarks_arr


def add_color(image, landmarks_arr, feature):
    """ Changes the color of a given feature eg. mouth, nose
    """

    overlay = image.copy()
    output = image.copy()

    (j, k) = FACIAL_LANDMARKS_IDXS[feature]
    pts = landmarks_arr[j:k]
    hull = cv2.convexHull(pts)
    cv2.drawContours(overlay, [hull], -1, COLORS[0], -1)

    cv2.addWeighted(overlay, 0.75, output, 1 - 0.75, 0, output)

    return output


def main():
    image_file = sys.argv[1]
    output_file = sys.argv[2]
    facial_feature = sys.argv[3]

    image = cv2.imread(image_file)

    landmarks_arr = get_landmarks(image)

    # Choose modification
    output = add_color(image, landmarks_arr, facial_feature)

    cv2.imshow("Image", output)
    cv2.waitKey(0)
    cv2.imwrite(output_file, output)


if __name__ == "__main__":
    main()
