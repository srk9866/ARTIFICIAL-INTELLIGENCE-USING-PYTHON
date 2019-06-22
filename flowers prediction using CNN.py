# -*- coding: utf-8 -*-
"""
Created on Sun Apr 15 12:40:58 2018

@author: Abhishek
"""

from keras.models import load_model
from keras.preprocessing import image
import numpy as np
import cv2
model = load_model('agrimodel.h5')
model.compile(loss='categorical_crossentropy',
                      optimizer='adam',
                      metrics=['accuracy'])
from skimage.io import imread
from skimage.transform import resize

def detect(frame):
    try:
        img = resize(frame,(64,64))
        img = np.expand_dims(img,axis=0)
        if(np.max(img)>1):
            img = img/255.0
        prediction = model.predict(img)
        print(prediction)
        p=np.argmax(prediction)
        
        if(p==0):
            print ("DAISY")
            return "daisy"
            print ("ROSE")
            return "rose"
        elif(p==1):
            print("DANDELION")
            return "dandelion"
        elif(p==2):
            print ("ROSE")
            return "rose"
        elif(p==3):
            print("SUNFLOWER")
            return "sunflower"
        else:
            print("TULIP")
            return "tulip"
    except AttributeError:
        print("shape not found")

frame=cv2.imread("sunflower.jpg")
data = detect(frame)
   
     
