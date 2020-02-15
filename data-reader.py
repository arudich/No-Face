import sys
sys.path.insert(0,"./TrainedNN")
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
    def __init__(self,datapoints=20):
        self.actors_file = "faceScrub/facescrub_actors.txt"
        self.actress_file = "faceScrub/facescrub_actresses.txt"
        self.datapoints = datapoints

        self.face_dict = {}
        self.read_data() # fills face_dict with {name,[url,url,...]}

        self.training_data = [] #[([one]],(96,96,3))...] 96x96 rgb images
        self.format_data() #turn dict into (96,96,3) images

    def format_data(self):
        # data = [] # [(name,url),...] pairs for randomized training order
        ref_dict = {} #person to one-hot encoding
        i = 0
        for key in self.face_dict:
            l = np.zeros(self.datapoints)
            l[i] = 1
            ref_dict[key] = l
            i += 1

        for (key,urls) in self.face_dict.items():
            for url in urls:
                data.append((key,url))

        random.shuffle(data)
        self.pull_images(data,ref_dict)
        
    def pull_images(self,data,ref_dict,batch_size=50):
        training = open("classifier_training_data.txt","w+")
        model = m.create_model()
        model.load_weights('TrainedNN/open_face.h5')
        i = 0
        batch_encodings = []
        batch_images = []
        alignment = Align.AlignDlib('TrainedNN/models/landmarks.dat')
        for (name,url) in data:
            img = self.url_to_image(url)
            if img is None:
                continue
            img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
            faces = alignment.getAllFaceBoundingBoxes(img)
            #face = alignment.getLargestFaceBoundingBox(img)
            if len(faces) != 1:
                if len(faces) == 0:
                    print("Not a face")
                else:
                    print("Too many faces")
                continue
            face = faces[0]

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
                batch_encodings.append(ref_dict[name])
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

    def get_face_from_bounding_box(self,face,img,alignment):
        x = face.left()
        y = face.top()
        w = face.width()
        h = face.height()
        img1 = cv2.rectangle(img ,(x,y),(x+w,y+h),(255, 0, 0), 2)
        # plt.subplot(131)
        # plt.imshow(img)
        # plt.subplot(132)
        # plt.imshow(img1)
        # plt.subplot(133)
        crop_img = img1[y:y+h, x:x+w]

        face_aligned = alignment.align(96, img, alignment.getLargestFaceBoundingBox(img), landmarkIndices=Align.AlignDlib.OUTER_EYES_AND_NOSE)
        #plt.imshow(face_aligned)
        #print(face_aligned.shape)
        #plt.show()
        return face_aligned

    def read_data(self):
        actor = open(self.actors_file,'r')
        actress = open(self.actress_file,'r')

        #first line is column headers
        actor.readline()
        actress.readline()

        index = 0
        actors_done,actress_done = False,False
        while True:
            if index%2 == 0:
                name,url = self.read_line(actor.readline())
            else:
                name,url = self.read_line(actress.readline())
            index += 1

            if name is None:
                continue

            if len(self.face_dict) == self.datapoints and not name in self.face_dict:
                if index%2 == 0: actors_done = True
                else: actress_done = True
                if(actors_done and actress_done): break
                else: continue

            l = self.face_dict.get(name,[])
            l.append(url)
            self.face_dict[name] = l

        actor.close()
        actress.close()

    def read_line(self,line):
        elems = line.split("\t")
        if len(elems) < 4:
            return None,None
        return elems[0],elems[3] #name,url

    def url_to_image(self,url):
        # download the image, convert it to a NumPy array, and then read
        # it into OpenCV format
        print(url)
        try:
            resp = urllib.request.urlopen(url)
        
            image = np.asarray(bytearray(resp.read()), dtype="uint8")
            image = cv2.imdecode(image, cv2.IMREAD_COLOR)
        except:
            print(":(")
            return None
        # return the image
        print("Working URL!")
        return image

if __name__ == '__main__':
    FD = FaceData(20)




