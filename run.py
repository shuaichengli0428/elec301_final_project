# wo shi sb abcabcabc heiheihei gangdu
# xiaolu
# import redis_queue as rq
# hahahahahah
# lue lue lue
# abc
# Test github SyncFork functionality
# Test github syncFork functionality 2
from keras.layers import Dense , Activation , Conv2D , MaxPooling2D , Flatten ,Dropout,Convolution2D
from keras.models import Sequential
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix , accuracy_score
from keras.preprocessing.image import ImageDataGenerator , array_to_img ,img_to_array , load_img
from keras.utils import to_categorical

#import cv2
import glob
import matplotlib.image as mpimg
import matplotlib.pyplot as plt

import matplotlib.gridspec as gridspec
import numpy as np
import pandas as pd
from PIL import Image
#import skimage.color as color
import sklearn.preprocessing as preprocessing
import sys

#from keras.datasets import mnist
#download mnist data and split into train and test sets
#(X_train, y_train), (X_test, y_test) = mnist.load_data()


# Load training data
training_data = pd.read_csv("train_sorted.csv",encoding = "latin1")

training_images = []
training_numbers = []
training_labels = []

image_stack_train=np.empty((0,268,182,3))
# Convert each image into an array
for (i, filename) in enumerate(glob.glob("C:/Users/apple/PycharmProjects/elec301_final_project/train_posters/*.jpg")):
    im_array_train = mpimg.imread(filename)
    im = img_to_array(im_array_train)[:, :, :]
    training_images.append(im)


# Encode y
y_train = training_data.Genre
y_train = to_categorical(y_train)

print(y_train)


X_images = np.array(training_images)
X_images = X_images.reshape(3094,268,182,3)


# Data Augmentation
data_aug = ImageDataGenerator(
    rotation_range=50,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode="nearest",)
#data_aug.fit(X_images)


#Defining model Architecture. Verify model accuracy
#create model
model = Sequential()
#add model layers
model.add(Conv2D(32, kernel_size=3, activation="relu", input_shape=(268,182,3)))
model.add(Conv2D(16, kernel_size=3, activation="relu"))
model.add(Flatten())
model.add(Dense(4, activation="softmax"))

model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

model.compile(optimizer="adam",loss="categorical_crossentropy",metrics=["accuracy"])
model.fit_generator(data_aug.flow(X_images,y_train, batch_size=64),epochs=30,verbose=1,steps_per_epoch=5)


#pred_test = model.predict(x_test)
