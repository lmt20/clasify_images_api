import cv2 as cv
import numpy as np
import os
import shutil


class sift_extraction:
    def __init__(self):
        self.sift = cv.xfeatures2d.SIFT_create()

    def extract(self, image):
        gray_image = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
        kp, des = self.sift.detectAndCompute(gray_image, None)
        return des

    def descriptionSize():
        return 128


def save_extracted_features(input_images_path, output_des_path, limit_nbest_keypoints=0):
    image_names = os.listdir(input_images_path)
    count = 0
    for image_name in image_names:
        image_path = os.path.join(input_images_path, image_name)
        image = cv.imread(image_path, 0)
        if limit_nbest_keypoints == 0:
            sift = cv.xfeatures2d.SIFT_create()
        else:
            sift = cv.xfeatures2d.SIFT_create(limit_nbest_keypoints)
        kp, des = sift.detectAndCompute(image, None)
        # save each file
        output_file_path = os.path.join(output_des_path, image_name)
        np.savez_compressed(output_file_path, des)
        count += 1
        print(f"image {count} extracted done!")


def load_extracted_features(des_path):
    des_files = os.listdir(des_path)
    des_file_path = os.path.join(des_path, des_files[0])
    des_npz = np.load(des_file_path)
    print(des_npz.files)
    print(des_npz['arr_0'].shape)


# Run extract feature
def run(input_images_path, output_des_path, limit_nbest_keypoints=0):
    if os.path.isdir(output_des_path):
        shutil.rmtree(output_des_path)
    os.mkdir(output_des_path)
    save_extracted_features(
        input_images_path, output_des_path, limit_nbest_keypoints)
    load_extracted_features(output_des_path)


# input_images_path = "/home/lmtruong1512/Pictures/Data/collapsed_animals"
# output_des_path = "/home/lmtruong1512/Codes/BTL_CSDLDPT/extracted_files/extracted_SIFT100"
# # clear ouput_des_path
# if os.path.isdir(output_des_path):
#     shutil.rmtree(output_des_path)
# os.mkdir(output_des_path)
# save_extracted_features(input_images_path, output_des_path)

# load description
# output_des_path = "/home/lmtruong1512/Codes/BTL_CSDLDPT/extracted_files"
# load_extracted_features(output_des_path)

# run by terminal
if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('images_path')
    parser.add_argument('descriptions_path')
    parser.add_argument('limit_nbest_keypoints', type=int)
    options = parser.parse_args()
    print(options)
    if options.limit_nbest_keypoints:
        run(options.images_path, options.descriptions_path,
            options.limit_nbest_keypoints)
    else:
        run(options.images_path, options.descriptions_path)
