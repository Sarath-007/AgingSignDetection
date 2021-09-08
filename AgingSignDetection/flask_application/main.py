import os
from app import app
import urllib.request
from flask import Flask, flash, request, redirect, url_for, render_template
from werkzeug.utils import secure_filename
from flask import Flask

from tensorflow import keras

import cv2
from matplotlib import pyplot
from matplotlib.patches import Rectangle
from matplotlib.patches import Circle
from mtcnn.mtcnn import MTCNN
from PIL import Image
from numpy import asarray

UPLOAD_FOLDER = 'static/uploads/'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024


wrinkle_model = keras.models.load_model('D:/Verzeo_Final_Project/flask_application/Models/Wrinkle_efficent_net')
eye_model=model = keras.models.load_model('D:/Verzeo_Final_Project/flask_application/Models/Puffed_Eyes_DL_efficent_net_Model')
dark_spots_model=keras.models.load_model('D:/Verzeo_Final_Project/flask_application/Models/darK_spots_efficent_net_model')


ALLOWED_EXTENSIONS = set(['png', 'jpg', 'jpeg', 'gif'])

class Preprocess_Image:
    

    def extract_eye(self,data, result_list):
        a={}
        pyplot.imshow(data)

        ax = pyplot.gca()

        for result in result_list:

            x, y, width, height = result['box']

            for key, value in result['keypoints'].items():
                a[key]=value
            #print(a)
            # create and draw dot
        #             dot = Circle(value, radius=30, color='red',fill=False)
        #             ax.add_patch(dot)
            i=data[y+20:a['mouth_right'][1]-15,x:a['mouth_right'][0]+20]

        #         cv2.imshow('img',i)
        #         cv2.waitKey(0)
        #         cv2.destroyAllWindows()
            i=cv2.resize(i,(224,224))
    # show the plot
        return i;
    def extract_face(pixels,detector, required_size=(224, 224)):
# load image from file
        #pixels = pyplot.imread(filename)
    # create the detector, using default weights
        
    # detect faces in the image
        results = detector.detect_faces(pixels)
    # extract the bounding box from the first face
        x1, y1, width, height = results[0]['box']
        x2, y2 = x1 + width, y1 + height
    # extract the face
        face = pixels[y1:y2, x1:x2]
    # resize pixels to the model size
        image = Image.fromarray(face)
        image = image.resize(required_size)
        face_array = asarray(image)
        return face_array
    def extract_wrinkles(self,file_location):

      img = cv2.imread(file_location)
      gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
      thresh = cv2.threshold(gray, 120, 255, cv2.THRESH_BINARY)[1]
      kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (4,4))
      dilate = cv2.morphologyEx(thresh, cv2.MORPH_DILATE, kernel)
      diff = cv2.absdiff(dilate, thresh)
      edges = 255 - diff
      return edges

    def puffed_eye(self,file_location,temp,temp2):

        detector = MTCNN()
        img = cv2.imread(file_location)
        i=cv2.resize(img,(236,354))
        # grey=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
        result = detector.detect_faces(i)
        print(result)
        keypoints = result[0]['keypoints']

        if temp==1:
            cv2.putText(i, 'DARK CIRCLES', (80, 30), cv2.FONT_HERSHEY_PLAIN, 1, (0, 0, 255), 1)
        if temp2==1:
            cv2.putText(i, 'PUFFY EYES', (40, 200), cv2.FONT_ITALIC, 0.4, (0, 255, 255), 1)
            cv2.putText(i, 'PUFFY EYES', (150, 200), cv2.FONT_ITALIC, 0.4, (0, 255, 255), 1)

         # cv2.putText(img, 'DARK CIRCLES', (80, 30), cv2.FONT_HERSHEY_PLAIN, 1, (0, 0, 255), 1)
         # cv2.putText(img, 'PUFFY EYES', (40, 200), cv2.FONT_ITALIC, 0.4, (0, 255, 255), 1)
        # cv2.putText(img, 'PUFFY EYES', (150, 200), cv2.FONT_ITALIC, 0.4, (0, 255, 255), 1)
        cv2.circle(i, (keypoints['left_eye']), 20, (0, 0, 255), 2)
        cv2.circle(i, (keypoints['right_eye']), 20, (0, 0, 255), 2)
        return i





def predict(image_location,filename):

      object=Preprocess_Image()
      image=cv2.imread(image_location)
      image=cv2.resize(image,(224,224))
      detector = MTCNN()
      faces = detector.detect_faces(image)
      try:
          eye=object.extract_eye(image,faces)
      except:
          eye=image
      detector = MTCNN()
      try:
          face=object.extract_face(image,detector)
      except:
          face=image
      eye=eye.reshape(1,eye.shape[0],eye.shape[1],eye.shape[2])
      face=face.reshape(1,face.shape[0],face.shape[1],face.shape[2])
      wrinkle = wrinkle_model.predict(face).argmax()
      puffed_eye = eye_model.predict(eye).argmax()
      dark_spots = dark_spots_model.predict(face).argmax()
      
      pred=[]
      if wrinkle==0:
          pred.append("The Person has no wrinkles")
      else:
          pred.append("The Person has wrinkles")
      if puffed_eye==0:
          pred.append("The Person doesn't have puffed eyes")
      else:
          pred.append("The Person has puffed eyes")
      if dark_spots==0:
          pred.append("The Person has no dark circles")
      else:
          pred.append("The Person has dark circles")
      wrinkle_face=object.extract_wrinkles(image_location)
      puffed_eye_face=object.puffed_eye(image_location,dark_spots,puffed_eye)
      if wrinkle==1:
          wrinkle_file_name=str(1)+filename
          cv2.imwrite(r'D:/Verzeo_Final_Project/flask_application/static/downloads/'+str(1)+filename,wrinkle_face)
      else:
          wrinkle_file_name=None
      if puffed_eye==1 or dark_spots==1:
          print("##########################################")
          puff_eye_file_name=str(2)+filename
          cv2.imwrite(r'D:/Verzeo_Final_Project/flask_application/static/downloads/'+str(2)+filename,puffed_eye_face)
      else:
          puff_eye_file_name=None


      
      return pred[0],pred[1],pred[2],wrinkle_file_name,puff_eye_file_name
 
    

	
@app.route('/')
def upload_form():
	return render_template('upload.html')

@app.route('/', methods=['GET','POST'])
def upload_predict():
    if request.method=='POST':
        image_file=request.files["image"]
        if image_file:
            
            image_location=os.path.join(app.config['UPLOAD_FOLDER'], image_file.filename)
            image_file.save(image_location)
            w,p,d,wl,pf=predict(image_location,image_file.filename)
            
            return render_template('upload.html',wrinkle=w,puffed_eye=p,dark_spots=d,image_loc=image_file.filename,output_image=wl,p_face=pf)
        return render_template('upload.html',prediction=0,data=1,image_loc=None,output_image=None,p_face=None)
    
    





if __name__ == "__main__":
    app.run(debug=True)