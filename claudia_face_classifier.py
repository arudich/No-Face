from face_classifier_training import FaceClassifierTrainer
import tensorflow as tf
import keras
from keras import layers
from keras.models import Model
from keras.regularizers import l2
import keras.backend as K
import keras.losses as losses
from keras.layers import Conv2D, ZeroPadding2D, Activation, Input, concatenate, Dense
from keras.layers.core import Lambda, Flatten, Dense
from keras.layers.normalization import BatchNormalization
from keras.layers.pooling import MaxPooling2D, AveragePooling2D
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
import matplotlib.pyplot as plt

class FaceClassifier(FaceClassifierTrainer):
    def __init__(self,num_people = 0,arch_num = 1):
        self.img_dir = "./images"
        if(num_people == 0):
            self.num_people = self.count_people()
            print("Counted ", self.num_people, " people")
            #input("If this is right, press enter. Otherwise cntrl-c outta here")
        else: self.num_people = num_people
        self.classifier = self.init_model(self.num_people)
        self.OF = m.create_model()
        self.ANN = self.init_ANN_arch(arch_num)
        self.person_dict = self.read_person_dict()

        self.init_classifier_gradients()

        self.Adam = tf.compat.v1.train.AdamOptimizer(learning_rate = .001)
        self.update = self.Adam.apply_gradients(zip(self.ANN_trainer,self.ANN.trainable_weights))
        #self.Adam_2 = tf.compat.v1.train.AdamOptimizer(learning_rate = .00025)
        #self.update_2 = self.Adam_2.apply_gradients(zip(self.proj_grads,self.ANN.trainable_weights))

        self.sess = tf.compat.v1.Session()
        init = tf.compat.v1.global_variables_initializer()
        self.sess.run(init)
        tf.compat.v1.keras.backend.set_session(self.sess)

        self.load_models()

        '''

        img = self.bound_img('./images/avi/avi_0.JPG')
        imgs = np.array([img])

        yTrue = np.zeros(self.num_people)
        yTrue[2] = 1
        yTrue = np.array([yTrue])

        feed_dict = {}
        self.forward_pass(imgs,yTrue,feed_dict)

        for i in range(10):
            feed_dict = {}
            self.forward_pass(imgs,yTrue,feed_dict)
            self.sess.run(self.update,feed_dict=feed_dict)
            print("Loss: ")
            self.sess.run(tf.print(self.loss),feed_dict=feed_dict)
            print("Norm: ")
            self.sess.run(tf.print(self.proj_size),feed_dict=feed_dict)
            #self.sess.run(self.update_2,feed_dict=feed_dict)

        plt.subplot(1,3,1)
        plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        plt.subplot(1,3,2)
        plt.imshow(cv2.cvtColor((img+self.ANN_predict_img(img))/255.0, cv2.COLOR_BGR2RGB))
        #print(img+self.ANN_predict_img(img))
        pred = self.predict_bounded_face(img+self.ANN_predict_img(img))
        classification = self.identify_pred(pred)
        print(pred)
        print(classification)
        plt.subplot(1,3,3)
        plt.imshow(cv2.cvtColor((self.ANN_predict_img(img))/255.0, cv2.COLOR_BGR2RGB))
        
        plt.show()
        '''



    def init_classifier_gradients(self):
        self.img = tf.compat.v1.placeholder(tf.float32,shape=(None,96,96,3))
        self.sum_images = tf.math.add(self.img,self.ANN.output)

        self.yTrue = tf.compat.v1.placeholder(tf.float32)
        self.loss = tf.multiply(-1.0,tf.losses.MSE(self.yTrue,self.classifier.output))
        self.loss_grads = tf.gradients(self.loss,self.classifier.output)
        self.classifier_grads = tf.gradients(self.classifier.output,
            self.classifier.input,grad_ys = self.loss_grads)
        self.OF_grads = tf.gradients(self.OF.output,self.OF.input,grad_ys = self.classifier_grads)

        self.proj_size = tf.square(1.0-tf.norm(tf.norm(self.ANN.output,axis=[1,2]),axis=1)*.0001)
        self.proj_size_grads = tf.gradients(self.proj_size,self.ANN.output)
        #self.ANN_mag_grads = tf.gradients(self.ANN.output,self.ANN.trainable_weights,grad_ys=self.proj_size_grads)

        self.grad_sum = tf.squeeze(tf.add(self.proj_size_grads,self.OF_grads),axis=0)
        self.ANN_trainer = tf.gradients(self.ANN.output,self.ANN.trainable_weights,grad_ys=self.grad_sum)


    def forward_pass(self,imgs,yTrue,feed_dict): 
        assert(np.shape(yTrue) == (1,self.num_people))
        assert(np.shape(imgs) == (1,96,96,3))
        #fills feed_dict for training step
        #returns projection_grads for training ANN
        #feed_dict = {}
        feed_dict[self.ANN.input] = imgs
        projection = self.ANN_predict_imgs(imgs)

        feed_dict[self.yTrue] = yTrue
        feed_dict[self.img] = imgs
        #feed_dict[self.projection] = projection

        img_sum = imgs + projection
        #print(img_sum)
        pred = self.OF_embed_bounded_faces(img_sum)
        feed_dict[self.OF.input] = img_sum
        feed_dict[self.classifier.input] = pred

        return feed_dict

    def load_models(self):
        self.load_model(self.classifier)
        self.OF.load_weights('TrainedNN/open_face.h5')

    def predict_face_from_cam(self,img_path):
        img = self.bound_img(img_path)
        pred = self.predict_bounded_face(img)
        print("Prediction: " + str(pred))
        classification = self.identify_pred(pred)
        print(classification)
        return pred,classification

    def OF_embed_bounded_faces(self,imgs):
        assert(np.shape(imgs[0]) == (96,96,3))
        return self.OF.predict(imgs)

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
        return cv2.imread(path,cv2.IMREAD_COLOR)

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

    def init_ANN_arch(self,arch_num):
        if arch_num == 1:
            return self.ANN_arch_1()
        else:
            print("No architecture with that number...using architecture 1")
            return self.ANN_arch_1()

    def ANN_predict_imgs(self,imgs):
        return self.ANN.predict(imgs)

    def ANN_predict_img(self,img):
        return self.ANN.predict(np.array([img]))[0]

    def ANN_arch_1(self):
        #print("Architecture 1 init:")
        myInput = Input(shape=(96, 96, 3))

        x = ZeroPadding2D(padding=(3, 3), input_shape=(96, 96, 3))(myInput)
        x = Conv2D(64, (7, 7), strides=(1, 1), name='conv1')(x) 
        x = BatchNormalization(axis=3, epsilon=0.00001, name='bn1')(x)
        # 64 7x7 convos
        #shape = (96,96,64)   
        #print("Conv 1 output shape = " + str(np.shape(x))) 
        y = ZeroPadding2D(padding = (2,2))(x)
        y = Conv2D(64, (5 , 5), strides = (1,1))(y)
        y = BatchNormalization(axis=3, epsilon=0.00001, name='bn2')(y)
        #shape = (96,96,64) 
        #print("Conv 2 output shape = " + str(np.shape(y))) 

        z = ZeroPadding2D(padding = (1,1))(y)
        z = Conv2D(64, (3 , 3), strides = (1,1))(z)
        z = BatchNormalization(axis=3, epsilon=0.00001, name='bn3')(z)
        #shape = (96,96,16)
        #print("Conv 3 output shape = " + str(np.shape(z))) 

        inception_3a_3x3 = Conv2D(96, (1, 1), name='inception_3a_3x3_conv1')(z)
        inception_3a_3x3 = BatchNormalization(axis=3, epsilon=0.00001, name='inception_3a_3x3_bn1')(inception_3a_3x3)
        inception_3a_3x3 = Activation('relu')(inception_3a_3x3)
        inception_3a_3x3 = ZeroPadding2D(padding=(1, 1))(inception_3a_3x3)
        inception_3a_3x3 = Conv2D(128, (3, 3), name='inception_3a_3x3_conv2')(inception_3a_3x3)
        inception_3a_3x3 = BatchNormalization(axis=3, epsilon=0.00001, name='inception_3a_3x3_bn2')(inception_3a_3x3)
        inception_3a_3x3 = Activation('relu')(inception_3a_3x3)

        inception_3a_5x5 = Conv2D(16, (1, 1), name='inception_3a_5x5_conv1')(z)
        inception_3a_5x5 = BatchNormalization(axis=3, epsilon=0.00001, name='inception_3a_5x5_bn1')(inception_3a_5x5)
        inception_3a_5x5 = Activation('relu')(inception_3a_5x5)
        inception_3a_5x5 = ZeroPadding2D(padding=(2, 2))(inception_3a_5x5)
        inception_3a_5x5 = Conv2D(32, (5, 5), name='inception_3a_5x5_conv2')(inception_3a_5x5)
        inception_3a_5x5 = BatchNormalization(axis=3, epsilon=0.00001, name='inception_3a_5x5_bn2')(inception_3a_5x5)
        inception_3a_5x5 = Activation('relu')(inception_3a_5x5)

        inception_3a_1x1 = Conv2D(64, (1, 1), name='inception_3a_1x1_conv')(x)
        inception_3a_1x1 = BatchNormalization(axis=3, epsilon=0.00001, name='inception_3a_1x1_bn')(inception_3a_1x1)
        inception_3a_1x1 = Activation('relu')(inception_3a_1x1)

        inception_3a = concatenate([inception_3a_3x3, inception_3a_5x5, inception_3a_1x1], axis=3)
        #shape = (96,96,224)
        #print("Inception 3a shape: " + str(np.shape(inception_3a)))

        # Inception3b
        inception_3b_3x3 = Conv2D(96, (1, 1), name='inception_3b_3x3_conv1')(inception_3a)
        inception_3b_3x3 = BatchNormalization(axis=3, epsilon=0.00001, name='inception_3b_3x3_bn1')(inception_3b_3x3)
        inception_3b_3x3 = Activation('relu')(inception_3b_3x3)
        inception_3b_3x3 = ZeroPadding2D(padding=(1, 1))(inception_3b_3x3)
        inception_3b_3x3 = Conv2D(32, (3, 3), name='inception_3b_3x3_conv2')(inception_3b_3x3)
        inception_3b_3x3 = BatchNormalization(axis=3, epsilon=0.00001, name='inception_3b_3x3_bn2')(inception_3b_3x3)
        inception_3b_3x3 = Activation('relu')(inception_3b_3x3)

        inception_3b_5x5 = Conv2D(32, (1, 1), name='inception_3b_5x5_conv1')(inception_3a)
        inception_3b_5x5 = BatchNormalization(axis=3, epsilon=0.00001, name='inception_3b_5x5_bn1')(inception_3b_5x5)
        inception_3b_5x5 = Activation('relu')(inception_3b_5x5)
        inception_3b_5x5 = ZeroPadding2D(padding=(2, 2))(inception_3b_5x5)
        inception_3b_5x5 = Conv2D(16, (5, 5), name='inception_3b_5x5_conv2')(inception_3b_5x5)
        inception_3b_5x5 = BatchNormalization(axis=3, epsilon=0.00001, name='inception_3b_5x5_bn2')(inception_3b_5x5)
        inception_3b_5x5 = Activation('relu')(inception_3b_5x5)

        inception_3b_1x1 = Conv2D(16, (1, 1), name='inception_3b_1x1_conv')(inception_3a)
        inception_3b_1x1 = BatchNormalization(axis=3, epsilon=0.00001, name='inception_3b_1x1_bn')(inception_3b_1x1)
        inception_3b_1x1 = Activation('relu')(inception_3b_1x1)

        inception_3b = concatenate([inception_3b_3x3, inception_3b_5x5, inception_3b_1x1], axis=3)
        #print("Inception 3b shape: " + str(np.shape(inception_3b)))
        #norm_O = BatchNormalization(axis=3, epsilon=0.00001, name='norm_out')(inception_3b)

        O = layers.Dense(3, activation='relu', kernel_regularizer=l2(0.0001))(inception_3b)
        #print("Output shape: " + str(np.shape(O)))

        model = Model(inputs=myInput,outputs=O)
        # model.compile("Adam",
        #     loss=self.custom_loss_function)
        return model

if __name__ == '__main__':
    FC = FaceClassifier()
    #FC.predict_face_from_cam('./images/avi/avi_0.JPG')