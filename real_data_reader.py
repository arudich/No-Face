import sys
sys.path.insert(0,"./TrainedNN")
import os
import TrainedNN.model as m
import TrainedNN.Align as Align
import TrainedNN.Data as Data
import TrainedNN.utils as utils
import numpy as np
import cv2
import urllib
import matplotlib.pyplot as plt
import random
from PIL import Image

data = []
class FaceData:
    def __init__(self):
        self.img_dir = "./images"
        self.encode_images()

        '''
        self.face_dict = {}
        self.read_data() # fills face_dict with {name,[url,url,...]}

        self.training_data = [] #[([one]],(96,96,3))...] 96x96 rgb images
        self.format_data() #turn dict into (96,96,3) images
        '''

    def encode_images(self):
        training = open("classifier_training_data.txt","w+")
        num_people = len(os.listdir(self.img_dir))-1
        img_index = 0

        model = m.create_model()
        model.load_weights('TrainedNN/open_face.h5')
        i = 0
        batch_size = 50
        batch_encodings = []
        batch_images = []
        alignment = Align.AlignDlib('TrainedNN/models/landmarks.dat')

        for directory in os.listdir(self.img_dir):
            if(directory == ".DS_Store"):
                continue
            encoding = np.zeros(num_people)
            encoding[img_index] = 1
            img_index += 1
            for i in range(110):
                path = os.path.join(self.img_dir,directory,directory + "_"+str(i) + ".jpg")
                img = self.load_jpg(path)
                face = alignment.getLargestFaceBoundingBox(img)
                if(face is None):
                    print("No face found :(")
                    continue

                face_aligned = self.get_face_from_bounding_box(face,img,alignment)   
                if face_aligned is None:
                    continue

                i += 1

                if i%batch_size == 0:
                    print("batch images")
                    print(np.array(batch_images).shape)
                    for img in batch_images:
                        print(np.array(img).shape)
                    predictions = model.predict(np.array(batch_images))
                    # print("predictions")
                    # print(predictions)
                    for (enc,pred) in zip(batch_encodings,predictions):
                        training.write("[")
                        for i in range(len(enc)-1):
                            training.write(str(enc[i]) + ",")
                        training.write(str(enc[-1]) + "]\t[")
                        for i in range(len(pred)-1):
                            training.write(str(pred[i]) + ",")
                        training.write(str(pred[-1]) + "]\n")
                        
                    batch_encodings = []
                    batch_images = []
                else:
                    print("Scanned image ", path)
                    batch_encodings.append(encoding)
                    batch_images.append(face_aligned)

        if len(batch_images) > 0:
            predictions = model.predict(np.array(batch_images))
            for (enc,pred) in zip(batch_encodings,predictions):
                training.write("[")
                for i in range(len(enc)-1):
                    training.write(str(enc[i]) + ",")
                training.write(str(enc[-1]) + "]\t[")
                for i in range(len(pred)-1):
                    training.write(str(pred[i]) + ",")
                training.write(str(pred[-1]) + "]\n")
            batch_encodings = []
            batch_images = []


        training.close()

                
              
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


if __name__ == '__main__':
    FD = FaceData()




