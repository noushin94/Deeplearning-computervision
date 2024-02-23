import numpy as np
import pandas as pd
import cv2
import glob
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
import joblib
import tensorflow
from tensorflow import keras
from keras import models, layers
from keras.utils import to_categorical
import matplotlib.pyplot as plt




features_vectors  = []
All_labels = []

for i,address in enumerate(glob.glob("/Users/noushinahmadvand/DocumentsCloud/myExercising/brain_Tumor- detection/*/*")):

     img = cv2.imread(address)
     img = cv2.resize(img, (32,32))
    
     img = img/255.0
     
     
     features_vectors.append(img)

     #print(address.split("/")[-2])
     labels = address.split("/")[-2]

     All_labels.append(labels)

     if i % 100 == 0 :
          print(f"[INFO] {i}/1000 processed")

features_vectors = np.array(features_vectors)          

Label_Encoder = LabelEncoder()

label_encoded =  Label_Encoder.fit_transform(All_labels)

label_encoded = to_categorical(label_encoded )




#print(label_encoded)

# spliting data into train and test

X_train , X_test , Y_train , Y_test = train_test_split(features_vectors, label_encoded, test_size= 0.2)

# defing the CNN network

net = models.Sequential([ 
     
                         layers.Conv2D(32, (3,3), activation = "relu", input_shape = (32,32,3)),
                         layers.MaxPooling2D((2,2)),
                         layers.Conv2D(32, (3,3), activation = "relu"),
                         layers.MaxPooling2D((2,2)),
                         layers.Flatten(),
                         layers.Dense(100, activation='relu'),
                         layers.Dense(2, activation= 'softmax')
     
     ])


print(net.summary())

net.compile(optimizer= "SGD",
          loss= "categorical_crossentropy",
        metrics = ["accuracy"])


h = net.fit(X_train,Y_train, batch_size=32, validation_data= (X_test, Y_test), epochs = 10)


loss , acc = net.evaluate(X_test, Y_test) # it by itself evaluate main y and compare it with y predict

net.save("CNN.h5")


plt.plot(h.history["accuracy"], label = "train accuracy" )
plt.plot(h.history["val_accuracy"], label = "test accuracy")
plt.plot(h.history["loss"], label = "train_accuracy" )
plt.plot(h.history["val_loss"], label = "test accuracy")
plt.legend()
plt.xlabel("epoch")
plt.ylabel("accuracy")
plt.title("tumor detection")
plt.show()





