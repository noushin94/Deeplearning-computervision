from tensorflow import keras
from keras import models
import cv2
import numpy as np



net2 = models.load_model("/Users/noushinahmadvand/Documents/myExercising/mlp1.h5")

img = cv2.imread("/Users/noushinahmadvand/Documents/myExercising/14 no.jpg")

# we nedd to do all preprocessing for our image here as well

#preprocesiing

r_img = cv2.resize(img, (32,32)).flatten()
r_img = r_img/255.0



#predicting with the model

out = net2.predict(np.array([r_img]))[0] # it predicted correctly, [0] first elements to be captured
 # to capture the result

out_label = np.argmax(out) # it gives us the indixe with max value among out elements
# creating a list to see the result of the out by "No" an "active"
category_name = ["No", "active"]

print("this patient has ",category_name[out_label] ,"tumor")

 # to show the result on the picture

text = category_name[out_label]

cv2.putText(img , text , (10,30), cv2.FONT_HERSHEY_PLAIN, 1.0, (0,255,0), 3)


#showing the image

cv2.imshow("image", img)
cv2.waitkey(0)



                  

