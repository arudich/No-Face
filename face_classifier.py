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
            self.num_people = self.count_people()
            print("Counted ", self.num_people, " people")
            #input("If this is right, press enter. Otherwise cntrl-c outta here")
        else: self.num_people = num_people
        self.classifier = self.init_model(self.num_people)
        self.OF = m.create_model()
        self.person_dict = self.read_person_dict()

        self.init_classifier_gradients()
        self.Adam = tf.compat.v1.train.AdamOptimizer(learning_rate = .001)
        self.update = self.Adam.apply_gradients(zip(self.classifier_trainer,self.classifier.trainable_weights))
        #self.run_classifier_gradients()

        self.sess = tf.compat.v1.Session()
        init = tf.compat.v1.global_variables_initializer()
        self.sess.run(init)
        #tf.compat.v1.keras.backend.set_session(self.sess)
        #K.set_session(self.sess)
        self.load_model(self.classifier)
        self.OF.load_weights('TrainedNN/open_face.h5')

        img = self.bound_img('./images/avi/avi_0.JPG')
        img = np.array([img])
        projection = np.random.uniform(size=(1,96,96,3))
        test = self.predict_bounded_face((img+projection)[0])

        yTrue = np.zeros(self.num_people)
        yTrue[2] = 1
        yTrue = np.array([yTrue])

        feed_dict = {}
        self.forward_pass(img,projection,yTrue,feed_dict)
        '''
        print("Projection: ")
        self.sess.run(tf.print(self.projection),feed_dict=feed_dict)
        print("Image: ")
        self.sess.run(tf.print(self.img),feed_dict=feed_dict)
        print("Combination: ")
        self.sess.run(tf.print(self.sum_images),feed_dict=feed_dict)
        print("Loss: ")
        self.sess.run(tf.print(self.loss),feed_dict=feed_dict)
        print("loss_grads: ")
        self.sess.run(tf.print(self.loss_grads),feed_dict=feed_dict)
        print("classifier_grads: ")
        self.sess.run(tf.print(self.classifier_grads),feed_dict=feed_dict)
        print("OF_grads: ")
        self.sess.run(tf.print(self.OF_grads),feed_dict=feed_dict)
        print("Projection_grads: ")
        self.sess.run(tf.print(self.projection_grads),feed_dict=feed_dict)
        '''
        print("Loss: ")
        self.sess.run(tf.print(self.loss),feed_dict=feed_dict)
        self.sess.run(tf.print(self.classifier_trainer),feed_dict=feed_dict)
        print("------------------------------------------------------------")
        self.sess.run(tf.print(self.classifier.trainable_weights),feed_dict=feed_dict)
        self.sess.run(self.update,feed_dict=feed_dict)
        print("Loss: ")
        self.sess.run(tf.print(self.loss),feed_dict=feed_dict)
        self.sess.run(self.update,feed_dict=feed_dict)
        print("Loss: ")
        self.sess.run(tf.print(self.loss),feed_dict=feed_dict)
        self.sess.run(self.update,feed_dict=feed_dict)
        print("Loss: ")
        self.sess.run(tf.print(self.loss),feed_dict=feed_dict)

    def init_classifier_gradients(self):
        # actor_inputs = {
        #     self.actor.input : states,
        #     self.critic.get_layer('states').input : states,
        #     self.critic.get_layer('actions').input : real_actions}
        # critic_inputs = {
        #     self.critic.get_layer('states').input : states,
        #     self.critic.get_layer('actions').input : actions,
        #     self.y_values : y_values
        # }
        # self.sess.run(self.updateCritic,feed_dict=critic_inputs)
        # self.sess.run(self.updateActor, feed_dict=actor_inputs)

        # self.actor_Adam = tf.train.AdamOptimizer(learning_rate = self.actor_lr)
        # self.critic_Adam = tf.train.AdamOptimizer(learning_rate = self.critic_lr)

        # self.value_grads = tf.gradients(self.critic.output, self.critic.get_layer("actions").input)

        # self.param_grads = tf.gradients(
        #     self.actor.output,self.actor.trainable_weights,grad_ys = tf.multiply(-1.,self.value_grads)) 

        # self.updateActor = self.actor_Adam.apply_gradients(zip(self.param_grads,self.actor.trainable_weights))

        # self.y_values = tf.placeholder(tf.float32)
        # self.critic_loss = tf.losses.mean_squared_error(self.y_values,self.critic.outputs)
        # self.updateCritic = self.critic_Adam.minimize(self.critic_loss,var_list = self.critic.trainable_weights)

        self.yTrue = tf.compat.v1.placeholder(tf.float32)
        self.loss = tf.losses.MSE(self.yTrue,self.classifier.output)
        self.loss_grads = tf.gradients(self.loss,self.classifier.output)
        self.classifier_grads = tf.gradients(self.classifier.output,
            self.classifier.input,grad_ys = self.loss_grads)
        self.classifier_trainer = tf.gradients(self.classifier.output,
            self.classifier.trainable_weights,grad_ys = self.loss_grads)
        self.OF_grads = tf.gradients(self.OF.output,self.OF.input,grad_ys = self.classifier_grads)
        self.img = tf.compat.v1.placeholder(tf.float32,shape=(None,96,96,3))
        self.projection = tf.compat.v1.placeholder(tf.float32,shape=(None,96,96,3))
        self.sum_images = tf.math.add(self.img,self.projection)
        self.projection_grads = tf.gradients(self.sum_images,self.projection,grad_ys = self.OF_grads)


    def forward_pass(self,img,projection,yTrue,feed_dict): 
        assert(np.shape(yTrue) == (1,self.num_people))
        assert(np.shape(img) == (1,96,96,3))
        assert(np.shape(projection) == (1,96,96,3))
        #fills feed_dict for training step
        #returns projection_grads for training ANN
        #feed_dict = {}
        feed_dict[self.yTrue] = yTrue
        feed_dict[self.img] = img
        feed_dict[self.projection] = projection

        img_sum = img + projection
        pred = self.OF_embed_bounded_faces(img_sum)
        feed_dict[self.OF.input] = img_sum
        feed_dict[self.classifier.input] = pred

        #return feed_dict

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

    def run_classifier_gradients(self):
        optimizer = tf.compat.v1.train.AdamOptimizer()
        yTrue = np.zeros(self.num_people)
        yTrue[2] = 1
        projections = tf.zeros((1,96,96,3))
        img = self.bound_img('./images/avi/avi_0.JPG')
        imgs = tf.convert_to_tensor(np.array([img]),dtype=tf.float32)

        print(self.classifier(tf.random.uniform((1,128))))
        exit(0)

        with tf.GradientTape() as tape:
            #print("Projection: ",projections)
            tape.watch(projections)
            OF_in = projections + imgs
            print("Combined Image: ", OF_in)
            OF_out = self.OF(OF_in)
            # tf.print(OF_out)
            # print("Open Face output: ", OF_out.eval())
            # print("Open Face output: ", tf.convert_to_tensor(OF_out).numpy())
            #print(tape.gradient(OF_out,projections))
            #exit(0)
            class_out = self.classifier.predict(OF_out)
            print("Classifier output: ", class_out)
            loss = tf.losses.MSE(np.array([yTrue]),class_out)
            print("Loss: ", loss)
            exit(0)
            grads = tape.gradient(loss, projection)
            print(grads)
            optimizer.apply_gradients(zip(grads, projection))
            print("Projection after step: ",projection)

if __name__ == '__main__':
    FC = FaceClassifier()
    #FC.predict_face_from_cam('./images/avi/avi_0.JPG')