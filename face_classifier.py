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
    def __init__(self,num_people = 0,arch_num = 1, num_pictures = 110):
        self.img_dir = "./images"
        self.num_pictures = 110
        self.arch_num = arch_num
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

        self.sess = tf.compat.v1.Session()
        init = tf.compat.v1.global_variables_initializer()
        self.sess.run(init)
        tf.compat.v1.keras.backend.set_session(self.sess)

        self.load_model(self.classifier)
        self.OF.load_weights('TrainedNN/open_face.h5')
        #self.load_all_models()

        self.load_ANN_model(name="ANN_Arch_"+str(arch_num) + "_Epoch_" + str(20))
        self.test_ANN()
        #self.train_ANN()
        #self.save_ANN_model(name="ANN_Arch_"+str(self.arch_num))


    def ANN_train_on_batch(self,indeces,batch_size=32):
        print("Starting Training Episode")
        batch_sample_indeces = random.sample(indeces,batch_size)
        batch_imgs = []
        batch_truth = []
        for (person,pic_num) in batch_sample_indeces:
            person_key = self.person_dict[person]
            img_path = self.img_dir+"/"+person_key+"/"+person_key+"_"+str(pic_num)+".jpg"
            img = self.bound_img(img_path)
            if img is None:
                continue

            yTrue = np.zeros(self.num_people)
            yTrue[person] = 1.0
            batch_imgs.append(img)
            batch_truth.append(yTrue)

        batch_imgs = np.array(batch_imgs)
        batch_truth = np.array(batch_truth)

        if(np.size(batch_imgs) == 0):
            print("All sampled picture invalid")
            return

        print("Running Forward Pass")
        feed_dict = self.forward_pass(batch_imgs,batch_truth)
        print("Updating")
        self.sess.run(self.update,feed_dict=feed_dict)
        #print("Loss: ")
        #self.sess.run(tf.print(self.loss),feed_dict=feed_dict)
        #print("Norm: ")
        #self.sess.run(tf.print(self.proj_size),feed_dict=feed_dict)


    def train_ANN(self):
        training_indeces = []
        test_index = int(self.num_pictures*2/3)

        for i in range(self.num_people):
            for j in range(test_index):
                training_indeces.append((i,j))

        for i in range(50):
            self.ANN_train_on_batch(training_indeces,batch_size=32)
            if(i%10 == 0):
                self.save_ANN_model(name="ANN_Arch_"+str(self.arch_num) + "_Epoch_" + str(i))
        


    def test_ANN(self,test_size=100):
        test_indeces = []
        test_index = int(self.num_pictures*2/3)

        for i in range(self.num_people):
            for j in range(test_index):
                test_indeces.append((i,j))

        batch_sample_indeces = random.sample(test_indeces,test_size)

        truths = []
        imgs = []
        observed_test_size = 0
        for (person,pic_num) in batch_sample_indeces:
            person_key = self.person_dict[person]
            img_path = self.img_dir+"/"+person_key+"/"+person_key+"_"+str(pic_num)+".jpg"
            img = self.bound_img(img_path)
            if img is None:
                continue
            observed_test_size += 1
            imgs.append(img)
            truths.append(person_key)

        imgs = np.array(imgs)
        projs = self.ANN.predict(imgs)

        OF_classifier_outs = self.OF.predict(imgs)
        classifier_preds = self.classifier.predict(OF_classifier_outs)

        OF_ANN_outs = self.OF.predict(imgs+projs)
        ANN_preds = self.classifier.predict(OF_ANN_outs)

        classifications = [self.person_dict[np.argmax(pred)] for pred in classifier_preds]
        ANN_classifications = [self.person_dict[np.argmax(pred)] for pred in ANN_preds]

        classification_acc = np.sum([a == b for (a,b) in zip(truths,classifications)])/observed_test_size
        ANN_acc = np.sum([a == b for (a,b) in zip(truths,ANN_classifications)])/observed_test_size
        print(classifications)
        print(ANN_classifications)
        print(truths)
        print("Classification Accuracy: ",classification_acc)
        print("Attacked Classification Accuracy: ", ANN_acc)

        '''
        for (person,pic_num) in batch_sample_indeces:
            person_key = self.person_dict[person]
            img_path = self.img_dir+"/"+person_key+"/"+person_key+"_"+str(pic_num)+".jpg"
            img = self.bound_img(img_path)
            if img is None:
                continue
            pred = self.predict_bounded_face(img)
            classification = self.identify_pred(pred)
            projection = self.ANN.predict(np.array([img]))[0]

            full_img = img + projection
            pred = self.predict_bounded_face(full_img)
            ANN_class = self.identify_pred(pred)
            self.show_pictures([img,projection,full_img],classification,ANN_class)
            '''

    def show_pictures(self,imgs,classification,ANN_class,div=True):
        num_imgs = len(imgs)
        for i in range(num_imgs):
            plt.subplot(1,num_imgs,i+1)
            img = imgs[i]
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = img/255.0 if div else img
            #print(img)
            #print(np.shape(img))
            plt.imshow(img)
            if(i == 0):
                plt.title(classification)
        plt.title(ANN_class)
        plt.show()

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


    def forward_pass(self,imgs,yTrue): 
        #fills feed_dict for training step
        #returns projection_grads for training ANN
        feed_dict = {}
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

    def load_all_models(self):
        self.load_model(self.classifier)
        self.OF.load_weights('TrainedNN/open_face.h5')
        self.load_ANN_model()

    def predict_face_from_cam(self,img_path):
        img = self.bound_img(img_path)
        if img is None: return None,"Could not find face"
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
        elif arch_num == 2:
            return self.ANN_arch_2()
        else:
            print("No architecture with that number...using architecture 1")
            return self.ANN_arch_1()

    def ANN_predict_imgs(self,imgs):
        return self.ANN.predict(imgs)

    def ANN_predict_img(self,img):
        return self.ANN.predict(np.array([img]))[0]

    def ANN_arch_2(self):
        dims = 96*96*3
        myInput = Input(shape=(96, 96, 3))

        f = Flatten()(myInput)
        d = layers.Dense(3000, activation='relu', kernel_regularizer=l2(0.0001))(f)
        d = layers.Dense(3000, activation='relu', kernel_regularizer=l2(0.0001))(d)
        d = layers.Dense(3000, activation='relu', kernel_regularizer=l2(0.0001))(d)
        d = layers.Dense(10000, activation='relu', kernel_regularizer=l2(0.0001))(d)
        d = layers.Dense(dims, activation='relu', kernel_regularizer=l2(0.0001))(d)
        O = layers.Reshape((96,96,3))(d)

        model = Model(inputs=myInput,outputs=O)
        # model.compile("Adam",
        #     loss=self.custom_loss_function)
        return model

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

    def save_ANN_model(self,save_dir = "models/",name="ANN_model"):
        save_name = save_dir + name
        self.ANN.save_weights(save_name)

    def load_ANN_model(self, load_dir = "models/",name="ANN_model"):
        load_name = load_dir + name
        self.ANN.load_weights(load_name)

if __name__ == '__main__':
    FC = FaceClassifier(arch_num=1)
    #FC.train_ANN()
    #FC.test_ANN()
    #FC.save_ANN_model()
    #FC.predict_face_from_cam('./images/avi/avi_0.JPG')