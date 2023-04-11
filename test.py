# test
import pandas as pd
import numpy as np
import csv
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.externals import joblib
#import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import make_classification
from keras.models import Sequential
from keras.layers import Dense, Activation, Conv2D, MaxPooling2D, Flatten
from keras.preprocessing.image import ImageDataGenerator
from keras.utils.np_utils import to_categorical
from PIL import Image
import os
import glob


train_data = pd.read_csv('train_sorted.csv',encoding = "latin1")
y_train = train_data.Genre
#y_train = y_train.values


img = Image.open("97033.jpg").convert("RGB")
img = img.resize((91,134),Image.ANTIALIAS)
arr = np.array(img)
flat_arr = arr.ravel()


#""""
path_train = glob.glob("C:/Users/apple/PycharmProjects/elec301_final_project/train_posters/*.jpg")
for poster in path_train:
    id = poster.strip(".jpg")
    file = Image.open(id + ".jpg").convert("RGB")
    file = file.resize((91,134),Image.ANTIALIAS)
    arr1 = np.array(file)
    flat1 = arr1.ravel()
    #np.reshape(flat_array,195104)
    flat_arr = np.vstack((flat_arr,flat1))

#print flat_arr 
#"""""



img_test = Image.open("84548.jpg").convert("RGB")
img_test = img_test.resize((91,134),Image.ANTIALIAS)
arr_test = np.array(img_test)
flat_arr_test = arr_test.ravel()



path_test = glob.glob("C:/Users/apple/PycharmProjects/elec301_final_project/test_posters/*.jpg")
for poster_test in path_test:
    id_test = poster_test.strip(".jpg")
    file_test = Image.open(id_test + ".jpg").convert("RGB")
    file_test = file_test.resize((91,134),Image.ANTIALIAS)
    arr_test = np.array(file_test)
    flat2 = arr_test.ravel()
    flat_arr_test = np.vstack((flat_arr_test,flat2))


""""
clf = RandomForestClassifier(n_estimators=100, max_depth=None,random_state=0)
clf.fit(flat_arr, y_train)
RandomForestClassifier(bootstrap=True, class_weight=None, criterion='gini',
            max_depth=None, max_features='auto', max_leaf_nodes=None,
            min_impurity_decrease=0.0, min_impurity_split=None,
            min_samples_leaf=1, min_samples_split=2,
            min_weight_fraction_leaf=0.0, n_estimators=100, n_jobs=None,
            oob_score=False, random_state=0, verbose=0, warm_start=False)
pred = clf.predict(flat_arr_test)
print pred
np.savetxt("random.csv", pred, delimiter=',')
"""""
""""
pipeline = make_pipeline(preprocessing.StandardScaler(),
                         RandomForestRegressor(n_estimators=100))

hyperparameters = {'randomforestregressor__max_features': ['auto', 'sqrt', 'log2'],
                   'randomforestregressor__max_depth': [None, 5, 3, 1]}

clf = GridSearchCV(pipeline, hyperparameters, cv=10)

clf.fit(flat_arr, y_train)

pred = clf.predict(flat_arr_test)

print pred
"""""
"""""

MLPClassifier(activation='relu', alpha=0.0001, batch_size='auto',
              beta_1=0.9, beta_2=0.999, early_stopping=False,
              epsilon=1e-08, hidden_layer_sizes=(100,),
              learning_rate='constant', learning_rate_init=0.001,
              max_iter=200, momentum=0.9, n_iter_no_change=10,
              nesterovs_momentum=True, power_t=0.5, random_state=1,
              shuffle=True, solver='adam', tol=0.0001,
              validation_fraction=0.1, verbose=False, warm_start=False)

clf = MLPClassifier(solver='adam', alpha=1e-5,hidden_layer_sizes=(100,), random_state=1)


clf.fit(flat_arr,y_train)
pred = clf.predict(flat_arr_test)
"""""

categorical_labels = to_categorical(y_train, num_classes=4)
print (categorical_labels)

data_aug = ImageDataGenerator(
    rotation_range=40,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode="nearest",)

model = Sequential()
model.add(Conv2D(64, kernel_size=3, activation="relu", input_shape=(268,182,3)))
model.add(Conv2D(32, kernel_size=3, activation="relu"))
model.add(Flatten())
model.add(Dense(10, activation="softmax"))

model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

#data_aug.fit(flat_arr)

model.fit_generator(data_aug.flow(flat_arr, y_train, batch_size=64), epochs=30, verbose=2)
#pred = model.predict(flat_arr_test)

#print (pred)



