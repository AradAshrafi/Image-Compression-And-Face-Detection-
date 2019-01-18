import cv2
import os

from lib.lib import DATASET_LOCATION


def load_images_from_folder():
    images = []
    for filename in os.listdir(DATASET_LOCATION):
        img = cv2.imread(os.path.join(DATASET_LOCATION, filename))
        if img is not None:
            images.append(img)
    return images
