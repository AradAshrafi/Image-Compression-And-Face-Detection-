from PIL import Image
import os
import numpy
import matplotlib.pyplot as plt
from scipy.linalg import eigh as largest_eigh
from lib.lib import DATASET_LOCATION, IMAGE_TO_COMPRESS_LOCATION


# load all images from folder path in gray scaled form
def load_dataset_images():
    images = []
    for filename in os.listdir(DATASET_LOCATION):
        path_to_file = os.path.join(DATASET_LOCATION, filename)
        # Convert Each Image to Gray Scale
        img = numpy.asarray(Image.open(path_to_file).convert("L"), dtype=numpy.float16)
        if img is not None:
            images.append(img)
    return numpy.asarray(images)


# load one image gray scaled
def load_image(path=IMAGE_TO_COMPRESS_LOCATION):
    img = numpy.asarray(Image.open(path).convert("L"), dtype=numpy.float64)
    return img


# calculates K largest eigenvalues and eigenvectors
# in our project K is equal to 6
def calculate_K_largest_eigs(square_matrix, K):
    N = len(square_matrix)
    evalues_large, evectors_large = largest_eigh(square_matrix, eigvals=(N - K, N - 1))
    return evalues_large, evectors_large


# calculates all eigenvalues and eigenvectors
def calculate_eigs(square_matrix):
    N = len(square_matrix)
    sorted_eigen_values, sorted_eigen_vectors = largest_eigh(square_matrix, eigvals=(0, N - 1))
    return sorted_eigen_values, sorted_eigen_vectors


# calculates average of all indices in a matrix
# note that indices can be a matrix either
def calculate_avg_matrix(images_matrix):
    avg_matrix = numpy.asarray(images_matrix[0].copy())
    for i in range(len(images_matrix) - 1):
        avg_matrix += images_matrix[i + 1]
    avg_matrix = avg_matrix / len(images_matrix)
    return avg_matrix


# calculate subtraction of image matrix from it's average matrix
def calculate_avg_subtraction(image_matrix, avg_matrix):
    subtracted_avg_matrix = image_matrix.copy()
    subtracted_avg_matrix -= avg_matrix
    return subtracted_avg_matrix


# calculate summation of matrix elements
def calculate_matrix_elements_summation(matrix):
    summation = 0
    for row in matrix:
        for element in row:
            summation += element
    return summation


def convert_k_diagonal_elements_to_zero(start_index, matrix):
    # clone original matrix,because in each iteration we change different diagonal values in it and we want original one
    compressed_matrix = matrix.copy()
    for i in range(len(matrix) - start_index):
        compressed_matrix[start_index + i][start_index + i] = 0
    return compressed_matrix


# plot compression values with K values
def draw_compression_plot(k_values, compression_rates):
    plt.plot(k_values, compression_rates)
    plt.show()
