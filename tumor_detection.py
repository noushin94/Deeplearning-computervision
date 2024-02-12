import numpy as np
import pandas as pd
import cv2
import glob
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
import joblib



features_vectors  = []
All_labels = []

for i,address in enumerate(glob.glob("/Users/noushinahmadvand/DocumentsCloud/myExercising/brain_Tumor- detection/*/*")):

     img = cv2.imread(address)
     img = cv2.resize(img, (32,32))
    
     img = img/255.0
     img = img.flatten()
     
     features_vectors.append(img)

     #print(address.split("/")[-2])
     labels = address.split("/")[-2]

     All_labels.append(labels)

     if i % 100 == 0 :
          print(f"[INFO] {i}/1000 processed")

features_vectors = np.array(features_vectors)          

#Label_Encoder = LabelEncoder()

#label_encoded =  Label_Encoder.fit_transform(All_labels)



X_train , X_test , Y_train , Y_test = train_test_split(features_vectors, All_labels, test_size= 0.2)

ml= KNeighborsClassifier(n_neighbors= 3)

ml.fit(X_train, Y_train)

accuracy = ml.score(X_test, Y_test)

print("accuracy: {:.2f}".format(accuracy*100))

joblib.dump(ml, 'kneigbour.pkl')