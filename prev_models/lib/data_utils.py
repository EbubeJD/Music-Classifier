import os
import pandas as pd
import numpy as np
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.utils import to_categorical
import shutil
from pathlib import Path

def all_images():
    '''moves all of the images from each genre folder to a large folder of all images to split the data up easily'''
    # defines source and destination folders
    source_folder = "Data/images_original"
    dest_folder = "Data/all_images"

    # make the folder if it doesn't already exist
    os.makedirs(dest_folder, exist_ok=True)

    # looks through each folder and moves the images from the individual folder to the all_images folder
    for genre in os.listdir(source_folder):
        genre_folder = os.path.join(source_folder, genre)
        if os.path.isdir(genre_folder):
            for filename in os.listdir(genre_folder):
                if filename.endswith(".png"):
                    src_path = os.path.join(genre_folder, filename)
                    dst_path = os.path.join(dest_folder, filename)
                    shutil.copyfile(src_path, dst_path)


def load_data(dataframe, IMG_DIR, IMG_SIZE):
    X, y = [], []
    for _, row in dataframe.iterrows():
        # links the name in the csv file to the name of the image in the image folder
        # img_path = os.path.join(IMG_DIR, row['filename'].replace('.wav', '.png'))
        original = row['filename']
        stem = ".".join(original.split(".")[:2])  # blues.00000
        image_name = stem.replace(".", "") + ".png"  # blues00000.png
        img_path = os.path.join(IMG_DIR, image_name)

        # if the image exists...
        if os.path.exists(img_path):

            # using tensorflow -> load image with a specific size and in black and white - using grayscale for faster training (can change later but not necessary)
            img = load_img(img_path, target_size=IMG_SIZE, color_mode='grayscale')
            # using tensorflow -> changes data from image format to array format and normalizes so all intensities of pixel are stored between 0 and 1
            img = img_to_array(img) / 255.0

            # adds image and corresponding label to X and y respectively
            X.append(img)
            y.append(row['label_idx'])
    
    # returns the array of images and a vector for y that will correspond with the genre it is (e.g. [0 0 1 0 0 ... 0 0 0] to match softmax output layer)
    return np.array(X), np.array(y)

