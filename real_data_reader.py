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

data = []
class FaceData:
    def __init__(self,num_people,add_single_person = False, person = "jacky",person_num = 7):
        self.img_dir = "./images"
        self.data_file = "classifier_training_data.txt"
        if(not add_single_person): #overwrite all people
            if(num_people == 0):
                num_people = self.count_people()
            print("Counted ", num_people, " people")
            input("If this is right, press enter. Otherwise cntrl-c outta here")
            self.encode_images(num_people,self.data_file)
        else: #append one person to txt file
            self.add_person(person,person_num,self.data_file)

    def count_people(self):
        num = 0
        print("Listing people:")
        for directory in os.listdir(self.img_dir):
            if(directory[0] == '.'):
                continue
            if(directory == "image_key.txt"):
                continue
            print(directory)
            num += 1
        return num

    def add_person(self,name,num,file):
        training = open(file,"a+")
        img_key = open("./images/image_key.txt","a+")
        img_key.write(str(num-1) + "\t" + name + "\t\n")

        model = m.create_model()
        model.load_weights('TrainedNN/open_face.h5')
        i = 0
        batch_size = 50
        batch_encodings = []
        batch_images = []
        alignment = Align.AlignDlib('TrainedNN/models/landmarks.dat')

        for i in range(110):
            path = os.path.join(self.img_dir,name,name + "_" + str(i) + ".jpg")
            img = self.load_jpg(path)
            face = alignment.getLargestFaceBoundingBox(img)
            if(face is None):
                print("No face found :(")
                continue

            face_aligned = self.get_face_from_bounding_box(face,img,alignment)   
            if face_aligned is None:
                continue

            i += 1

            print("Scanned image ", path)
            batch_encodings.append(num-1)
            batch_images.append(face_aligned) 

            if i%batch_size == 0:
                print("batching images")
                self.batch_predictions(training,batch_images,batch_encodings,model)
                    
                batch_encodings = []
                batch_images = []
                

        if len(batch_images) > 0:
            self.batch_predictions(training,batch_images,batch_encodings,model)

        training.close()
        image_key.close()

    def encode_images(self,num_people,file):
        training = open(file,"w+")
        img_key = open("./images/image_key.txt","w+")
        img_index = 0

        model = m.create_model()
        model.load_weights('TrainedNN/open_face.h5')
        i = 0
        batch_size = 50
        batch_encodings = []
        batch_images = []
        alignment = Align.AlignDlib('TrainedNN/models/landmarks.dat')


        for directory in os.listdir(self.img_dir):
            if(directory[0] == '.'):
                continue

            for i in range(110):
                path = os.path.join(self.img_dir,directory,directory + "_" + str(i) + ".jpg")
                img = self.load_jpg(path)
                face = alignment.getLargestFaceBoundingBox(img)
                if(face is None):
                    print("No face found :(")
                    continue

                face_aligned = self.get_face_from_bounding_box(face,img,alignment)   
                if face_aligned is None:
                    continue

                i += 1

                print("Scanned image ", path)
                batch_encodings.append(img_index)
                batch_images.append(face_aligned)

                if i%batch_size == 0:
                    print("batching images")
                    self.batch_predictions(training,batch_images,batch_encodings,model)
                        
                    batch_encodings = []
                    batch_images = []
                    
            img_key.write(str(img_index) + "\t" + directory + "\t\n")
            img_index += 1

        if len(batch_images) > 0:
            print("Final batch")
            self.batch_predictions(training,batch_images,batch_encodings,model)

        training.close()
        image_key.close()

    def batch_predictions(self,file,batch_images,batch_encodings,model):
        predictions = model.predict(np.array(batch_images))
        for (enc,pred) in zip(batch_encodings,predictions):
            file.write(str(enc) + "\t[")
            for i in range(len(pred)-1):
                file.write(str(pred[i]) + ",")
            file.write(str(pred[-1]) + "]\n")
                  
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
    append = False
    name = ''
    num = 0
    if(len(sys.argv) == 4):
        if(sys.argv[1] == "-append"):
            append = True
            name = sys.argv[2]
            num = int(sys.argv[3])

    FD = FaceData(0,append,name,num) #7 means append avi as the 7th person




