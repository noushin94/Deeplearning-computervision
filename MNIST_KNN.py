import numpy as np
import pandas as np
from sklearn.model_selection import train_test_split
import cv2
import tensorflow
from tensorflow import keras
from keras.datasets import mnist


(x_train,y_train) , (x_test, Y_test) = mnist.load_data()



for img in x_train:


    img = cv2.resize(img, (32,32))
