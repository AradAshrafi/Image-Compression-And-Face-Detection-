from PIL import Image
import os
import numpy
from scipy.linalg import eigh as largest_eigh
from lib.lib import DATASET_LOCATION


# load all images from folder path
def load_images_from_folder():
    images = []
    for filename in os.listdir(DATASET_LOCATION):
        path_to_file = os.path.join(DATASET_LOCATION, filename)
        # Convert Each Image to Gray Scale
        img = numpy.asarray(Image.open(path_to_file).convert("L"), dtype=numpy.float16)
        if img is not None:
            images.append(img)
    return numpy.asarray(images)


# load one image
def load_image(path):
    img = numpy.asarray(Image.open(path).convert("L"), dtype=numpy.float16)


def calculate_K_largest_eigs(symmetric_matrix, K):
    N = len(symmetric_matrix)
    evals_large, evecs_large = largest_eigh(symmetric_matrix, eigvals=(N - K, N - 1))
    return evals_large, evecs_large
