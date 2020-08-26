import cv2 as cv
import numpy as np

# img = np.array([[100.,102.], [20.,22.], [30.,30.]])
# print(img.shape)
# print(type(img))
# # img = cv.imread('/home/lmtruong1512/codes/BTL1/feature_extraction/img1.jpg', 0)
# # print(img.shape)
# # print(type(img))
# img_resize = cv.resize(img, (0,0), fx=2,fy=2    )
# print(img_resize.shape)
# print(img)
# print(img_resize)

# print(np.zeros((5,5)))
image = np.array([[1,2,3],[4,5,6],[7,8,9]])
gradient_kernel_x = np.array([[-1, 0, 1]])
gradient_kernel_y = np.reshape(gradient_kernel_x, (3,1))
gradient_x = cv.filter2D(image,-1,gradient_kernel_x)
gradient_y = cv.filter2D(image,-1,gradient_kernel_y)

print(gradient_x)

