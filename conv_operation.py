import cv2
import numpy as np

img = cv2.imread("/Users/noushinahmadvand/Documents/myExercising/Lenna_(test_image).png")

# for vretical edges
kernel = np.array([[-1],[1]])
#for horizontal edges
kernel2 = np.array([[1,-1]])

# to apply this kernel to ourimage
#edge detection 

out_img = cv2.filter2D(img, cv2.CV_8U, kernel)
out_img2 = cv2.filter2D(img, cv2.CV_8U, kernel2)


cv2.imshow("image", img)
cv2.imshow("output", out_img)
cv2.imshow("output2", out_img2)
cv2.waitKey(0)
cv2.destroyAllWindows()
