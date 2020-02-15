import tensorflow as tf
from keras import layers
from keras.models import Model
from keras.regularizers import l2
import keras.backend as K
import keras.losses as losses
import string
import numpy as np

class FaceClassifier:
    def __init__(self):
        self.model = self.init_model()
        self.train_model()

    def load_model(self,save_dir = "models/",name = "model"):
        load_name = save_dir + name
        self.model.load_weights(load_name)

    def save_model(self,save_dir = "models/",name="classifier_model"):
        save_name = save_dir + name
        self.model.save_weights(save_name)

    def init_model(self):
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
        O = layers.Dense(20, activation='softmax', kernel_regularizer=l2(0.0001))(d2)
        model = Model(inputs=I,outputs=O)
        model.compile("Adam",
            loss="categorical_crossentropy",
            metrics = ['accuracy'])
        return model

    def train_model(self, file = "classifier_validation_data.txt"):
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

        acc = 0
        while acc < .95:
            self.model.fit(x=train_input,y=train_enc,epochs=10)
            test_acc = self.model.evaluate(x=train_input,y=train_enc)
            acc = test_acc[1]
            print("test accuracy = " + str(acc))

        out = self.model.evaluate(x=validation_input,y=validation_enc)
        print(self.model.metrics_names)
        print(out)
        self.save_model()



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
        return ret


    def classify(self,data):
        pass

if __name__ == '__main__':
    FC = FaceClassifier()

    