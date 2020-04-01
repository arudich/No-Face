from claudia_face_classifier import FaceClassifier


if __name__ == '__main__':
    fc = FaceClassifier()
    fc.predict_face_from_cam('./images/avi/avi_0.JPG')

    #This method runs a couple others
    '''
    def predict_face_from_cam(self,img_path):
        img = self.bound_img(img_path)
        pred = self.predict_bounded_face(img)
        print("Prediction: " + str(pred))
        classification = self.identify_pred(pred)
        print(classification)
        return pred,classification
    '''

    #bounded and warped images that go into OpenFace have shape (96,96,3)
    #if you want to do work on (96,96,3) warped images, you can use the methods
    #shown in the predict_face_from_cam method

    #predict_bounded_face runs a bounded face through OpenFace and the classifier
    #it outputs probabilities
    #identify_pred(probabilities) maps it to the name of the person

    #look at ./images/img_key.txt to see who the indeces correspond to 
    #in the output probabilities
