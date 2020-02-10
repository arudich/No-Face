import sys
sys.path.insert(0,"./TrainedNN")
import TrainedNN.model as m
import keras as K
import TrainedNN.Align as Align
import TrainedNN.Data as Data
import TrainedNN.utils as utils
import numpy as np
import cv2
import matplotlib.pyplot as plt


img = cv2.imread("two_people.jpg")
img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
alignment = Align.AlignDlib('TrainedNN/models/landmarks.dat')
#face = alignment.getLargestFaceBoundingBox(img)
faces = alignment.getAllFaceBoundingBoxes(img)

for face in faces:
    x = face.left()
    y = face.top()
    w = face.width()
    h = face.height()
    img1 = cv2.rectangle(img ,(x,y),(x+w,y+h),(255, 0, 0), 2)
    plt.subplot(131)
    plt.imshow(img)
    plt.subplot(132)
    plt.imshow(img1)
    plt.subplot(133)
    crop_img = img1[y:y+h, x:x+w]
    #print(np.array(crop_img).shape)
    #plt.imshow(crop_img)

    face_aligned = alignment.align(96, img, alignment.getLargestFaceBoundingBox(img), landmarkIndices=Align.AlignDlib.OUTER_EYES_AND_NOSE)
    plt.imshow(face_aligned)
    print(face_aligned.shape)
    plt.show()

exit(0)
model = m.create_model()
model.load_weights('TrainedNN/open_face.h5')

print(model.predict(np.array([face_aligned]),verbose=0))


