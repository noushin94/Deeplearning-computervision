import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import cv2
import tensorflow
from tensorflow import keras
from keras.datasets import mnist
from sklearn.neighbors import KNeighborsClassifier


(x_train,y_train) , (x_test, y_test) = mnist.load_data()

x_train_r = []

x_test_r = []


for img in x_train:


    img = cv2.resize(img, (32,32))
    img = img.flatten()/255.0
    x_train_r.append(img)

for img in x_test:

    img = cv2.resize(img, (32,32))
    img = img.flatten()/255.0
    x_test_r.append(img)


x_train_f =np.array(x_train_r)
x_test_f = np.array(x_test_r)

model = KNeighborsClassifier(n_neighbors= 5, weights= "uniform")


model.fit(x_train_f, y_train)

y_predict = model.predict(x_test_f)
