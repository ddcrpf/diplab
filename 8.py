# %%
import cv2 as cv

import matplotlib.pyplot as plt

import numpy as n

def convertToRGB(img):

    return cv.cvtColor(img, cv.COLOR_BGR2RGB)

face_cascade = cv.CascadeClassifier(cv.data.haarcascades + 'haarcascade_frontalface_default.xml')

image=cv.imread('images/image8.jfif')

grayimage=cv.cvtColor(image,cv.COLOR_BGR2GRAY)

faces=face_cascade.detectMultiScale(grayimage,scaleFactor=1.11,minNeighbors=5)

for (x, y, w, h) in faces:

    cv.rectangle(image, (x, y), (x+w, y+h), (0, 255, 0), 1)

    plt.imshow(convertToRGB(image))

    print('no of faces:'),print(len(faces))

# %%


# %%



