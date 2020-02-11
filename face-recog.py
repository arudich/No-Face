import sys
sys.path.insert(0,"./TrainedNN")
import TrainedNN.model as m
import keras as K
import TrainedNN.Align as Align
import TrainedNN.Data as Data
import TrainedNN.utils as utils
import numpy as np
import cv2
import urllib
import matplotlib.pyplot as plt

def url_to_image(url):
    # download the image, convert it to a NumPy array, and then read
    # it into OpenCV format
    try:
        resp = urllib.request.urlopen(url)
    except:
        return None
    print("success!")
    image = np.asarray(bytearray(resp.read()), dtype="uint8")
    image = cv2.imdecode(image, cv2.IMREAD_COLOR)
    # return the image
    return image


#img = cv2.imread("two_people.jpg")
url = "http://images1.fanpop.com/images/photos/1600000/adam-brody-3-adam-brody-1607233-814-1201.jpg"
img = url_to_image(url)

img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
alignment = Align.AlignDlib('TrainedNN/models/landmarks.dat')
face = alignment.getLargestFaceBoundingBox(img)
#faces = alignment.getAllFaceBoundingBoxes(img)

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


