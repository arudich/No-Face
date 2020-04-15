from claudia_face_classifier import FaceClassifier


if __name__ == '__main__':
    fc = FaceClassifier()

    ### Random shapes
    ### Darken nose area
    ### Eyebrow shape

    ### Lip color ###
    # fc.predict_face_from_cam('./images/claudia/claudia_4.JPG')
    # fc.predict_face_from_cam('./images/claudia/claudia_edited_lip_2.JPG')

    # recognizes dtom as avi!
    # fc.predict_face_from_cam('./images/dtom/dtom_83.JPG')
    # fc.predict_face_from_cam('./images/dtom/dtom_edited_lip.JPG')
    # fc.predict_face_from_cam('./images/dtom/dtom_4.JPG')
    # fc.predict_face_from_cam('./images/dtom/dtom_edited_lip_2.JPG')

    ### Moustache ###
    # fc.predict_face_from_cam('./images/claudia/claudia_72.JPG')
    # fc.predict_face_from_cam('./images/claudia/claudia_edited_facialhair.JPG')
    # fc.predict_face_from_cam('./images/aggie/aggie_97.JPG')
    # fc.predict_face_from_cam('./images/aggie/aggie_edited_facialhair.JPG')

    # probability of dtom increased
    # fc.predict_face_from_cam('./images/dtom/dtom_64.JPG')
    # fc.predict_face_from_cam('./images/dtom/dtom_edited_facialhair.JPG')

    # fc.predict_face_from_cam('./images/avi/avi_91.JPG')
    # fc.predict_face_from_cam('./images/avi/avi_edited_facialhair.JPG')

    ### Side facial hair ###
    # probability of avi increased
    # fc.predict_face_from_cam('./images/avi/avi_3.JPG')
    # fc.predict_face_from_cam('./images/avi/avi_edited_lighten.JPG')

    # fc.predict_face_from_cam('./images/aggie/aggie_1.JPG')
    # fc.predict_face_from_cam('./images/aggie/aggie_edited_lighten.JPG')

    # probability of aggie increased
    # fc.predict_face_from_cam('./images/dtom/dtom_24.JPG')
    # fc.predict_face_from_cam('./images/dtom/dtom_edited_lighten.JPG')

    ### Nose color ###
    # fc.predict_face_from_cam('./images/claudia/claudia_10.JPG')
    fc.predict_face_from_cam('./images/claudia/claudia_edited_nose.JPG')


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
