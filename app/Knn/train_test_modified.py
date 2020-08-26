import time
import math
import random
import os
import sys
import numpy as np
import cv2 as cv
from skimage import io
from sklearn.neighbors import KNeighborsClassifier
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from feature_extraction import sift_extraction
from Extract_BOW import extract_bow

n_neighbors = 5
#1. Prepare data: Dataset include 3 types of animals: Buom, Voi, Ga 
# (Each breed of animal include 150 images)--> total: 450 image
#2. Split data into trainset and testset with ratio: 8:2 
#3. In trainning set: each image assign label (Buom, Voi or Ga) 
# and feature vector of image --- each image represent === path of image
#4. Create knn model from feature vectors(SIFT->normalize BoW) below
#5. From each image in test set --> predict k  the most similar images ->
# therefor predict the label of the image. From result true or false -> calculate 
# accurate of the prediction
#6. Create a funtion (input: a image) -> (output: the most similar image) ->
#use knn with n_neighbor=1

#construct list of (path_img, catagory) and split it into trainset and testset

def split_data(dir_path, ratio):
    #constuct array of image paths [('path','Buom'),...]
    image_paths = []
    catagory_names = os.listdir(dir_path)
    for catagory_name in catagory_names:
        image_names = os.listdir(os.path.join(dir_path, catagory_name))
        for image_name in image_names:
            image_path = os.path.join(dir_path, catagory_name, image_name)
            image_paths.append((image_path, catagory_name))
    #split image into trainset and testset
    random.shuffle(image_paths)
    partition = int(len(image_paths)*ratio)
    train_set = image_paths[:partition]
    test_set = image_paths[partition:]
    return (train_set, test_set)

#calculate encode (SIFT-> BoW) of a image
def extract_encode_image (img_path, centroids):
    extract_encode = extract_bow.extract_bow(centroids)
    img = io.imread(img_path)
    # img = cv.imread(img_path)
    img_encode = extract_encode.extract(img)
    return img_encode

#cal img_encodes of trainset image {"img_path" : (img_encode, catagory), ...}
def cal_encodes_tranningset(train_set, centroids):
    encode_label_of_images = {}
    if not os.path.exists(os.path.join("app","file_encode_label.npy")):
        num = 0
        for (img_path, catagory) in train_set:
            img_encode = extract_encode_image(img_path, centroids)
            encode_label_of_images[img_path] = (img_encode, catagory)
            num += 1
            print("encoding image:", num, "completed")
        np.save(os.path.join("app","file_encode_label.npy"), encode_label_of_images)
    else:
        encode_label_of_images = np.load(os.path.join("app","file_encode_label.npy"),allow_pickle='TRUE').item()
    return encode_label_of_images

def construct_model_knn(encode_label_of_images):
    print("Begin training")
    clf = KNeighborsClassifier(n_neighbors=n_neighbors, weights='distance')
    trainX = [encode_label_of_images[path_image][0] for path_image in encode_label_of_images.keys()]
    trainY = [encode_label_of_images[path_image][1] for path_image in encode_label_of_images.keys()]
    clf.fit(trainX, trainY)
    print("Training completed")
    return clf

def find_similar_images(img_path, clf, centroids):
    image_encode= extract_encode_image(img_path, centroids)
    pre = clf.predict([image_encode])

    print(img_path, "- Predict: ", pre)
    (distant, nearest_img_index) = clf.kneighbors([image_encode])
    return (nearest_img_index, pre)

def cal_accuracy(test_set, clf, centroids):
    right_result = 0
    num = 0
    for (img_path, catagory) in test_set:
        image_encode = extract_encode_image(img_path, centroids)
        pre = clf.predict([image_encode])
        if pre == catagory:
            right_result+= 1
        num += 1
        print(num, ".", img_path, "-", catagory, ".Predict: ", pre)    
    accurate = right_result/(len(test_set))
    return accurate

def cal_final_accuracy(dir_path, path_centroids, ratio=0.8, num_iters=100):
    print('begin:')
    centroids = np.load(path_centroids)
    final_acc = 0
    count = 0
    for i in range(0,num_iters):
        (train_set, test_set) = split_data(dir_path, ratio)
        encode_label_of_images = cal_encodes_tranningset(train_set, centroids)
        clf = construct_model_knn(encode_label_of_images)
        accurate = cal_accuracy(test_set, clf, centroids)
        print("Accuracy", accurate)
        final_acc += accurate
        count+=1
    # print("final accuracy:", final_acc/count)
    return final_acc/count

def most_similar_image_demo_display(img_path, dir_path, path_centroids, ratio=1):
    print('begin:')
    centroids = np.load(path_centroids)
    (train_set, test_set) = split_data(dir_path, ratio)
    encode_label_of_images = cal_encodes_tranningset(train_set, centroids)
    clf = construct_model_knn(encode_label_of_images)
    nearest_img_index, pre = find_similar_images(img_path, clf, centroids)
    nearest_img_index = nearest_img_index[0]
    print(nearest_img_index)
    nearest_img_paths =[ list(encode_label_of_images.keys())[index] for index in nearest_img_index]
    print("Similar image: Path -", nearest_img_paths)
    print("Real Catagory: ",[encode_label_of_images[nearest_img_path][1] 
    for nearest_img_path in nearest_img_paths])
    # display result
    fig = plt.figure()
    img1 = fig.add_subplot(1,2,1)
    imgplot = plt.imshow(cv.imread(img_path))
    img1.set_title("Input Image")
    img1 = fig.add_subplot(1,2,2)
    imgplot = plt.imshow(cv.imread(nearest_img_paths[0]))
    img1.set_title("Most similar image: " +  encode_label_of_images[nearest_img_paths[0]][1])
    plt.show()

    print('begin:')
def demo_display(img_path, dir_path, path_centroids, ratio=1):
    centroids = np.load(path_centroids)
    (train_set, test_set) = split_data(dir_path, ratio)
    encode_label_of_images = cal_encodes_tranningset(train_set, centroids)
    clf = construct_model_knn(encode_label_of_images)
    nearest_img_index, pre = find_similar_images(img_path, clf, centroids)
    nearest_img_index = nearest_img_index[0]
    print(nearest_img_index)
    nearest_img_paths =[ list(encode_label_of_images.keys())[index] for index in nearest_img_index]
    print("Similar image: Path -", nearest_img_paths)
    print("Real Catagory: ",[encode_label_of_images[nearest_img_path][1] 
    for nearest_img_path in nearest_img_paths])
    fig = plt.figure(figsize=(20, 20))
    sub = fig.add_subplot(4, n_neighbors, int(n_neighbors/2)+1)
    sub.axis('off')
    sub.set_title("Input Image")
    sub.set_xticks([])
    sub.set_yticks([])
    plt.imshow(cv.imread(img_path))
    sub = fig.add_subplot(4, n_neighbors, int(n_neighbors/2) +n_neighbors + 1)
    # sub.set_title("Find " + str(n_neighbors) + " most similar images")
    sub.text(-0.7, 0.5,"Find " + str(n_neighbors) + " most similar images",
     ha='center', va='bottom', transform=sub.transAxes, style="italic", color="brown", size=12)
    sub.set_xticks([])
    sub.set_yticks([])
    sub.axis('off')
    plt.imshow(cv.imread("/home/lmtruong1512/Pictures/compare.png"))
    for index, path in enumerate(nearest_img_paths):
        sub = fig.add_subplot(4, n_neighbors, n_neighbors*2+index+1)
        sub.axis('off')
        sub.set_title(encode_label_of_images[path][1], style="oblique", size=11)
        sub.set_xticks([])
        sub.set_yticks([])
        plt.imshow(cv.imread(path))
    sub = fig.add_subplot(4, n_neighbors, n_neighbors*3 + int(n_neighbors/2) + 1)
    sub.axis('off')
    sub.set_xticks([])
    sub.set_yticks([])
    sub.text(-1, 0.4, '=> Predict:',
    ha='center', va='bottom', transform=sub.transAxes, style="italic", color="brown", size=12)
    plt.imshow(cv.imread(img_path))
    sub.text(0.5, -0.3, pre[0],
    ha='center', va='bottom', transform=sub.transAxes, style="italic", color="blue", size=15)
    plt.imshow(cv.imread(img_path))
    plt.show()

def find_similar_image_paths(img_path, dir_path, path_centroids, ratio=1):
    centroids = np.load(path_centroids)
    (train_set, test_set) = split_data(dir_path, ratio)
    encode_label_of_images = cal_encodes_tranningset(train_set, centroids)
    clf = construct_model_knn(encode_label_of_images)
    nearest_img_index, pre = find_similar_images(img_path, clf, centroids)
    nearest_img_index = nearest_img_index[0]
    print(nearest_img_index)
    nearest_img_paths =[ list(encode_label_of_images.keys())[index] for index in nearest_img_index]
    print("Similar image: Path -", nearest_img_paths)
    return nearest_img_paths

# if __name__ == '__main__':
#     import argparse
#     parser = argparse.ArgumentParser()
#     parser.add_argument('img_path')
#     options = parser.parse_args()
#     run(options.img_path)

# img_path = "/home/lmtruong1512/codes/BTL1/image_data/animals_test/vet.jpg"
# dir_path = "/home/lmtruong1512/codes/BTL1/image_data/animals_collection"
# path_centroids = "/home/lmtruong1512/codes/BTL1/centroid_files/sift_centroids.npy"


def run(img_path):
    dir_path = os.path.join("app", "image_data", "animals")
    path_centroids = os.path.join("app", "centroid_files", "sift_centroids.npy")
    # most_similar_image_demo_display(img_path, dir_path, path_centroids)
    return find_similar_image_paths(img_path, dir_path, path_centroids, ratio=1)
# cal_final_accuracy(dir_path, path_centroids, num_iters=10)
# most_similar_image_demo_display(img_path, dir_path, path_centroids)
# demo_display(img_path, dir_path, path_centroids)

# run("/home/lmtruong1512/codes/Personal_Project/API_Classify_Image/image_data/animals_test/voi.jpg")