import cv2
import numpy as np

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense,Convolution2D,Dropout,Activation,Flatten,MaxPooling2D,Conv2D
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import to_categorical

from tensorflow.keras.models import load_model
from keras.preprocessing.image import ImageDataGenerator

#Training directory
data_dir=r'train'

#Total classes(0 and O were made the same class because they look the same)
categories=['0','1','2','3','4','5','6','7','8','9','A','B','C','D','E','F','G','H','I','J','K','L','M','N','O','P','Q','R','S','T','U','V','W','X','Y','Z']
batch_size=16


#Defining CNN architecture
model=Sequential()
model.add(Conv2D(32,(3,3),input_shape=(64,64,1),activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Conv2D(32,(3,3),activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Flatten())
model.add(Dense(units=128,activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(units=36,activation='softmax'))
model.compile(optimizer='adam',loss='sparse_categorical_crossentropy',metrics=['accuracy'])

print(model.summary())

#Defining training and validation sets
train_datagen=ImageDataGenerator(rescale = 1./255,
                                   shear_range = 0.2,
                                   zoom_range = 0.2,
                                   horizontal_flip = False, validation_split=0.2)

train_generator = train_datagen.flow_from_directory(
    data_dir,
    batch_size=batch_size,
    target_size = (64, 64),
    color_mode='grayscale',
    class_mode='sparse',
    subset='training') # set as training data

validation_generator = train_datagen.flow_from_directory(
    data_dir, # same directory as training data
    target_size = (64, 64),
    color_mode='grayscale',
    batch_size=batch_size,
    class_mode='sparse',
    subset='validation') # set as validation data

#Training
model.fit_generator(
    train_generator,
    steps_per_epoch = 3000,
    validation_data = validation_generator,
    validation_steps = 1000,
    epochs = 20)

#Saving model
model.save('model.h5')
