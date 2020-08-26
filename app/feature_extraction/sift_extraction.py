import cv2 as cv
import numpy as np
import os
import shutil

def extract_sift(image):
    gray_image = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    sift_extractor = cv.xfeatures2d.SIFT_create()
    key, des = sift_extractor.detectAndCompute(gray_image, None)
    return des

def save_extracted_features(input_images_path, output_des_path):
    image_names = os.listdir(input_images_path)
    count = 0
    for image_name in image_names:
        image_path = os.path.join(input_images_path, image_name)
        image = cv.imread(image_path)
        des = extract_sift(image)
        # save the descriptor to file
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
def run(input_images_path, output_des_path):
    if os.path.isdir(output_des_path):
        shutil.rmtree(output_des_path)
    os.mkdir(output_des_path)
    save_extracted_features(input_images_path, output_des_path)
    load_extracted_features(output_des_path)


# input_images_path = "/home/lmtruong1512/Pictures/Data/collapsed_animals"
# output_des_path = "/home/lmtruong1512/codes/BTL1/extracted_files/extracted_SIFT100"

# run by terminal
if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('images_path')
    parser.add_argument('descriptions_path')
    options = parser.parse_args()
    print(options)
    run(options.images_path, options.descriptions_path)
# /usr/bin/python3 /home/lmtruong1512/codes/BTL1/feature_extraction/sift_extraction.py /home/lmtruong1512/Pictures/Data/collapsed_animals /home/lmtruong1512/TestSIFT
