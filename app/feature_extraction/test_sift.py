import numpy as np
import cv2 as cv
img = cv.imread('/home/lmtruong1512/codes/BTL1/feature_extraction/img1.jpg')
gray= cv.cvtColor(img,cv.COLOR_BGR2GRAY)
sift = cv.SIFT_create(10000)
kp, des = sift.detectAndCompute(gray,None)
imga=cv.drawKeypoints(img,kp,img,flags=cv.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
cv.imwrite('sift_keypoints.jpg',imga)

print(des.shape)
print(des)