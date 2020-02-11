import tensorflow as tf
from keras import layers
from keras.models import Model
from keras.regularizers import l2
import keras.backend as K

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
        h3 = layers.Dense(150, activation='relu', kernel_regularizer=l2(0.0001))(h2)
        h4 = layers.Dense(500, activation='relu', kernel_regularizer=l2(0.0001))(h3)
        h5 = layers.BatchNormalization()(h4)
        O = layers.Dense(20, activation='softmax', kernel_regularizer=l2(0.0001))(h5)
        model = Model(inputs=I,outputs=O)
        return model

    def train_model(self, file = "classifier_training_data.txt"):
        f = open(file,"r")
        while True:
            line = f.readline()
            if line == []: break
            contents = line.split("\t")
            print(contents)
            encoding = contents[0]
            data = contents[1]


    def classify(self,data):
        pass

if __name__ == '__main__':
    FC = FaceClassifier()

    