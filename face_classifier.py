from face_classifier_training import FaceClassifierTrainer
import tensorflow as tf
from keras import layers
from keras.models import Model
from keras.regularizers import l2
import keras.backend as K
import keras.losses as losses
import string
import numpy as np
import math
import random
import os
import sys
import cv2
sys.path.insert(0,"./TrainedNN")
import TrainedNN.model as m
import TrainedNN.Align as Align
import TrainedNN.Data as Data
import TrainedNN.utils as utils

class FaceClassifier(FaceClassifierTrainer):
    def __init__(self,num_people = 0):
        self.img_dir = "./images"
        if(num_people == 0):
            num_people = self.count_people()
            print("Counted ", num_people, " people")
            #input("If this is right, press enter. Otherwise cntrl-c outta here")
        self.classifier = self.init_model(num_people)
        self.load_model(self.classifier)
        self.OF = m.create_model()
        self.OF.load_weights('TrainedNN/open_face.h5')
        self.person_dict = self.read_person_dict()

        pred,classification = self.predict_face_from_cam('./images/avi/avi_0.jpg')

    def predict_face_from_cam(self,img_path):
        img = self.bound_img(img_path)
        pred = self.predict_bounded_face(img)
        print("Prediction: " + str(pred))
        classification = self.identify_pred(pred)
        print(classification)
        return pred,classification

    def predict_bounded_face(self,img):
        assert(np.shape(img) == (96,96,3))
        OF_out = self.OF.predict(np.array([img]))
        return self.classifier.predict(OF_out)[0]

    def identify_pred(self,prediction):
        return self.person_dict[np.argmax(prediction)]

    def bound_img(self,path):
        img = self.load_jpg(path)
        alignment = Align.AlignDlib('TrainedNN/models/landmarks.dat')
        face = alignment.getLargestFaceBoundingBox(img)
        if(face is None):
            print("No face found :(")
            return None

        face_aligned = self.get_face_from_bounding_box(face,img,alignment)   

        if face_aligned is None:
            print("No bounding box found :(")
            return None

        return face_aligned
            
    def load_jpg(self,path): 
        return cv2.imread(path)

    def get_face_from_bounding_box(self,face,img,alignment):
        x = face.left()
        y = face.top()
        w = face.width()
        h = face.height()
        img1 = cv2.rectangle(img ,(x,y),(x+w,y+h),(255, 0, 0), 2)

        crop_img = img1[y:y+h, x:x+w]

        face_aligned = alignment.align(96, img, alignment.getLargestFaceBoundingBox(img), landmarkIndices=Align.AlignDlib.OUTER_EYES_AND_NOSE)

        return face_aligned

    def read_person_dict(self):
        key = open("./images/image_key.txt","r")
        Lines = key.readlines()
        person_dict = {}
        for line in Lines:
            num,name,_ = line.split("\t")
            person_dict[int(num)] = name
        return person_dict

if __name__ == '__main__':
    FC = FaceClassifier()