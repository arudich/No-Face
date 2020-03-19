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

class FaceClassifierTrainer:
    def __init__(self,num_people = 0):
        self.img_dir = "./images"
        if(num_people == 0):
            num_people = self.count_people()
            print("Counted ", num_people, " people")
            input("If this is right, press enter. Otherwise cntrl-c outta here")

        self.model = self.init_model(num_people)
        self.train_model(num_people)
        #self.analyze_distances()
        #self.analyze_OF_closeness()

    def load_model(self,model,save_dir = "models/",name = "classifier_model"):
        load_name = save_dir + name
        model.load_weights(load_name)

    def save_model(self,save_dir = "models/",name="classifier_model"):
        save_name = save_dir + name
        self.model.save_weights(save_name)

    def init_model(self,num_people):
        I = layers.Input(shape=(128,), name='classifier_input')
        h1 = layers.Dense(500, activation='relu', kernel_regularizer=l2(0.0001))(I)
        h2 = layers.BatchNormalization()(h1)
        d1 = layers.Dropout(.2)(h2)
        h3 = layers.Dense(500, activation='relu', kernel_regularizer=l2(0.0001))(d1)
        h4 = layers.BatchNormalization()(h3)
        d2 = layers.Dropout(.2)(h4)
        # h5 = layers.Dense(150, activation='relu', kernel_regularizer=l2(0.0001))(d2)
        # h6 = layers.BatchNormalization()(h5)
        # d3 = layers.Dropout(.2)(h6)
        # h7 = layers.Dense(150, activation='relu', kernel_regularizer=l2(0.0001))(d3)
        # h8 = layers.BatchNormalization()(h7)
        # d4 = layers.Dropout(.2)(h8)
        O = layers.Dense(num_people, activation='softmax', kernel_regularizer=l2(0.0001))(d2)
        model = Model(inputs=I,outputs=O)
        model.compile("Adam",
            loss="categorical_crossentropy",
            metrics = ['accuracy'])
        return model

    def train_model(self, num_people, file = "classifier_validation_data.txt"):
        f = open(file,"r")
        encodings = []
        inputs = []
        while True:
            line = f.readline()
            if line == '': break
            contents = line.split("\t")
            encoding = int(contents[0])
            tmp = np.zeros(num_people)
            tmp[encoding] = 1
            encoding = tmp #convert index to one-hot encoding
            data = self.convert_str_to_list(contents[1])
            inputs.append(data)
            encodings.append(encoding)

        shuff = list(zip(encodings,inputs))
        random.shuffle(shuff)
        encodings,inputs = [np.array([i for i,j in shuff]),
                            np.array([j for i,j in shuff])]
        
        training_size = int(len(encodings)*2/3) 

        train_input = np.array(inputs[0:training_size])
        train_enc = np.array(encodings[0:training_size])

        validation_input = np.array(inputs[training_size:-1])
        validation_enc = np.array(encodings[training_size:-1])

        acc = 0
        while acc < .97:
            self.model.fit(x=train_input,y=train_enc,epochs=10)
            test_acc = self.model.evaluate(x=train_input,y=train_enc)
            acc = test_acc[1]
            print("test accuracy = " + str(acc))

        out = self.model.evaluate(x=validation_input,y=validation_enc)
        print(self.model.metrics_names)
        print(out)
        self.save_model()

    def analyze_OF_closeness(self,file = "classifier_training_data.txt"):
        f = open(file,"r")
        encodings = []
        inputs = []
        people = {}
        while True:
            line = f.readline()
            if line == '': break
            contents = line.split("\t")
            encoding = self.convert_str_to_list(contents[0])
            data = np.array(self.convert_str_to_list(contents[1]))
            person = np.argmax(encoding)
            l = people.get(person,[])
            l.append(data)
            people[person] = l

        for key1,value1 in people.items():
            for key2,value2 in people.items():
                total_diff = 0
                diffs = []
                for inp1 in value1:
                    for inp2 in value2:
                        diffs.append(np.linalg.norm(inp1-inp2))
                print("key1 = ",key1,"; key2 = ",key2)
                print(np.mean(diffs))
                print(np.std(diffs))
                print("--------------------")


    def analyze_distances(self, file = "classifier_training_data.txt"):
        f = open(file,"r")
        encodings = []
        inputs = []
        while True:
            line = f.readline()
            if line == '': break
            contents = line.split("\t")
            encoding = self.convert_str_to_list(contents[0])
            data = self.convert_str_to_list(contents[1])
            inputs.append(data)
            encodings.append(encoding)

        training_size = int(len(encodings)*2/3) 

        train_input = np.array(inputs[0:training_size])
        train_enc = np.array(encodings[0:training_size])

        validation_input = np.array(inputs[training_size:-1])
        validation_enc = np.array(encodings[training_size:-1])

        correct = 0
        confidence = 0
        i = 0
        for inp,enc in zip(validation_input,validation_enc):
            means = self.smallest_mean(train_input,train_enc,inp,len(enc))

            # num_neighbors = 100
            # nearest_neighbors,_ = self.knn(train_input,train_enc,inp,num_neighbors)
            # #print(nearest_neighbors)
            # neighbor_probabilities = np.sum(nearest_neighbors,axis=0)/num_neighbors
            # print(neighbor_probabilities)
            print(means)
            choice = np.argmax(means)
            print(choice)
            if(enc[choice] == 1):
                correct += 1
                print("correct")
            else: print("incorrect")
            i+= 1
            print(i)

        print(correct)
        print(confidence)
        print(correct/len(validation_input))



    def smallest_mean(self,train_inputs,train_encodings,test_input,num_people):
        dists = []
        for i in range(num_people):
            dists.append([])
        for inp,enc in zip(train_inputs,train_encodings):
            dist = np.linalg.norm(inp-test_input)
            dists[np.argmax(enc)].append(dist)

        means = []
        for d in dists:
            means.append(np.mean(d))

        return means

    def knn(self,train_inputs,train_encodings,test_input,k):
        nearest_dist = [math.inf] * k
        nearest_name = [[]] * k

        for inp,enc in zip(train_inputs,train_encodings):
            dist = np.linalg.norm(inp-test_input)
            max_dist_index = np.argmax(nearest_dist)
            max_dist = nearest_dist[max_dist_index]
            if max_dist > dist:
                nearest_dist[max_dist_index] = dist
                nearest_name[max_dist_index] = enc
        return nearest_name,nearest_dist


    def convert_str_to_list(self,string):
        #remove brackets and \n
        s = string.strip('[]\n')
        #convert remaining string to list of floats
        ret = []
        for f in s.split(","):
            if f == '':
                ret.append(0.)
            else:
                ret.append(float(f))
        return np.array(ret)

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

    def classify(self,data):
        pass

if __name__ == '__main__':
    FC = FaceClassifierTrainer() #defaults to counting the number of people who have directories

    