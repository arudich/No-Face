import tensorflow as tf
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
sys.path.insert(0,"./TrainedNN")
import TrainedNN.model as m
import TrainedNN.Align as Align
import TrainedNN.Data as Data
import TrainedNN.utils as utils
from face_classifier import FaceClassifier


class ANN:
    def __init__(self,arch_num):
        self.arch_num = arch_num
        self.model = self.init_arch(arch_num)

        fc = FaceClassifier()
        self.init_classifier_gradients(fc)
        self.init_model_grads(fc)

        self.Adam = tf.compat.v1.train.AdamOptimizer(learning_rate = .001)
        self.update = self.Adam.apply_gradients(zip(self.model_trainer,self.model.trainable_weights))

        self.sess = tf.compat.v1.Session()
        init = tf.compat.v1.global_variables_initializer()
        self.sess.run(init)

        fc.load_models()

        img = fc.bound_img('./images/avi/avi_0.JPG')
        imgs = np.array([img])
        projection = self.predict_imgs(imgs)

        yTrue = np.zeros(fc.num_people)
        yTrue[2] = 1
        yTrue = np.array([yTrue])

        feed_dict = {}
        fc.forward_pass(img,projection,yTrue,feed_dict)
        feed_dict[self.model.input] = imgs

        print("Loss: ")
        self.sess.run(tf.print(fc.loss),feed_dict=feed_dict)
        self.sess.run(self.update,feed_dict=feed_dict)
        print("Loss: ")
        self.sess.run(tf.print(fc.loss),feed_dict=feed_dict)
        self.sess.run(self.update,feed_dict=feed_dict)
        print("Loss: ")
        self.sess.run(tf.print(fc.loss),feed_dict=feed_dict)
        self.sess.run(self.update,feed_dict=feed_dict)
        print("Loss: ")
        self.sess.run(tf.print(fc.loss),feed_dict=feed_dict)
        

    def custom_loss_function(self,y_actual,y_pred):
        return 0

    def init_model_grads(self,fc):
        self.model_trainer = tf.gradients(self.model.output,
            self.model.trainable_weights,grad_ys = self.projection_grads)

    def init_arch(self,arch_num):
        if arch_num == 1:
            return self.arch_1()
        else:
            print("No architecture with that number...using architecture 1")
            return self.arch_1()

    def predict_imgs(self,imgs):
        return self.model.predict(imgs)

    def predict_img(self,img):
        return self.model.predict(np.array([img]))[0]

    def arch_1(self):
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

        O = layers.Dense(3, activation='relu', kernel_regularizer=l2(0.0001))(inception_3b)
        #print("Output shape: " + str(np.shape(O)))

        model = Model(inputs=myInput,outputs=O)
        # model.compile("Adam",
        #     loss=self.custom_loss_function)
        return model

    def init_classifier_gradients(self,fc):
        self.yTrue = tf.compat.v1.placeholder(tf.float32)
        self.loss = tf.losses.MSE(self.yTrue,fc.classifier.output)
        self.loss_grads = tf.gradients(self.loss,fc.classifier.output)
        self.classifier_grads = tf.gradients(fc.classifier.output,
            fc.classifier.input,grad_ys = self.loss_grads)
        self.classifier_trainer = tf.gradients(fc.classifier.output,
            fc.classifier.trainable_weights,grad_ys = self.loss_grads)
        self.OF_grads = tf.gradients(fc.OF.output,fc.OF.input,grad_ys = self.classifier_grads)
        self.img = tf.compat.v1.placeholder(tf.float32,shape=(None,96,96,3))
        self.projection = tf.compat.v1.placeholder(tf.float32,shape=(None,96,96,3))
        self.sum_images = tf.math.add(self.img,self.projection)
        self.projection_grads = tf.gradients(self.sum_images,self.projection,grad_ys = self.OF_grads)

if __name__ == '__main__':
    ann = ANN(1)


