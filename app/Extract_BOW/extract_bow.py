from sklearn.neighbors import KNeighborsClassifier
import cv2 as cv
import numpy as np
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from feature_extraction import sift_extraction


class extract_bow:
    def __init__(self, centroids):
        self.centroids = centroids
        self.knn = KNeighborsClassifier(n_neighbors=1, weights="distance")
        self.knn.fit(centroids, range(len(centroids)))

    def extract(self, img):
        des = sift_extraction.extract_sift(img)
        try:
            index = []
            for i, arr in enumerate(des):
                if np.any(np.isnan(arr)):
                    index.append(i)
            des = np.delete(des, index, axis=0)
            pred = self.knn.predict(des)
        except:
            length = 128
            des = np.zeros((1, length))
            pred = self.knn.predict(des)

        arr_count = np.zeros(len(self.centroids))
        for x in pred:
            arr_count[x] += 1
        return arr_count / len(des)


# centroids = np.load(
#     "/home/lmtruong1512/codes/BTL1/centroid_files/sift100_centroids128.npy")
# img = cv.imread(
#     "/home/lmtruong1512/codes/BTL1/image_data/animals_test/OIF-e2bexWrojgtQnAPPcUfOWQ.jpeg")
# extract_BoW = extract_bow(centroids)
# arr_bow = extract_BoW.extract(img)
# print(arr_bow)
