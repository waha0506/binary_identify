import cv2                 
import numpy as np         
import os                  
import glob                
import keras
import sys
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Activation, Dropout, Flatten, Dense, BatchNormalization
from keras import backend as K
from keras.models import model_from_json
from keras.models import load_model
from keras.preprocessing import image
from keras import backend as K
from keras.preprocessing.image import img_to_array, load_img
import tensorflow as tf   

config = tf.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.9
config.gpu_options.allow_growth = True
session = tf.Session(config=config)

img_width, img_height = 200, 200
input_shape = (img_width, img_height, 3) 
chanDim = -1

model = Sequential()
model.add(Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same', input_shape=input_shape))
model.add(MaxPooling2D((2, 2)))
model.add(Dropout(0.2))
model.add(Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same'))
model.add(MaxPooling2D((2, 2)))
model.add(Dropout(0.2))
model.add(Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same'))
model.add(MaxPooling2D((2, 2)))
model.add(Dropout(0.2))
model.add(Flatten())
model.add(Dense(128, activation='relu', kernel_initializer='he_uniform'))
model.add(Dropout(0.5))
model.add(Dense(2, activation='sigmoid'))

img = cv2.imread(sys.argv[1])
cv2.imshow('img', img)

green_img = np.full((84, 84, 3), 0, np.uint8)
red_img = np.full((84, 84, 3), 0, np.uint8)
full_layer = np.full((84, 84), 255, np.uint8)
green_img[:, :, 1] = full_layer
red_img[:, :, 2] = full_layer
pic=np.zeros(shape=(10,5,84,84,3))

for x in range(10):
    for y in range(5):
        crop_img = img[y*84:y*84+84, x*84:x*84+84]
        crop_img = cv2.resize(crop_img, (int(img_height),int(img_width)))
        model.load_weights('./model_saved.h5')
        arr = np.array(crop_img).reshape((img_width,img_height,3))
        arr = np.expand_dims(arr, axis=0)
        prediction = model.predict(arr)[0]
        print(prediction)
        print(x, y)
        if prediction[0] > 0.5:
            pic[x][y]=green_img
        else:
            pic[x][y]=red_img

        cv2.imshow('%s, %s'%(str(x),str(y)), crop_img)


cv2.imshow('grid', np.vstack([
    np.hstack([pic[0][0], pic[1][0], pic[2][0], pic[3][0], pic[4][0], pic[5][0], pic[6][0], pic[7][0], pic[8][0], pic[9][0]]), 
    np.hstack([pic[0][1], pic[1][1], pic[2][1], pic[3][1], pic[4][1], pic[5][1], pic[6][1], pic[7][1], pic[8][1], pic[9][1]]),
    np.hstack([pic[0][2], pic[1][2], pic[2][2], pic[3][2], pic[4][2], pic[5][2], pic[6][2], pic[7][2], pic[8][2], pic[9][2]]),
    np.hstack([pic[0][3], pic[1][3], pic[2][3], pic[3][3], pic[4][3], pic[5][3], pic[6][3], pic[7][3], pic[8][3], pic[9][3]]),
    np.hstack([pic[0][4], pic[1][4], pic[2][4], pic[3][4], pic[4][4], pic[5][4], pic[6][4], pic[7][4], pic[8][4], pic[9][4]]),
]))

cv2.waitKey()



