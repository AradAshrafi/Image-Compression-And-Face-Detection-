from PIL import Image
import os
import numpy
from scipy.linalg import eigh as largest_eigh
from lib.lib import DATASET_LOCATION, IMAGE_TO_COMPRESS_LOCATION


# load all images from folder path in gray scaled form
def load_images_from_folder():
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


def calculate_K_largest_eigs(square_matrix, K):
    N = len(square_matrix)
    evalues_large, evectors_large = largest_eigh(square_matrix, eigvals=(N - K, N - 1))
    return evalues_large, evectors_large


def calculate_eigs(square_matrix):
    N = len(square_matrix)
    sorted_eigen_values, sorted_eigen_vectors = largest_eigh(square_matrix, eigvals=(0, N - 1))
    return sorted_eigen_values, sorted_eigen_vectors


def SVD_decomposition(matrix):
    AtA = numpy.matmul(matrix.transpose(), matrix)
    # finding eigenvalues and eigenvectors of matrix
    sorted_eigenvalues, sorted_eigenvectors = calculate_eigs(square_matrix=AtA)
    # Build sigma matrix from AtA eigenvalues
    sigma = __build_sigma_matrix(sorted_eigenvalues=sorted_eigenvalues, matrix=matrix)
    # building V matrix
    # V is 320 * 320
    V = numpy.transpose(numpy.transpose(sorted_eigenvectors)[::-1])

    # building U matrix
    U = __build_U_matrix(matrix=matrix, sigma=sigma, V=V)
    return U, sigma, V


def __build_sigma_matrix(sorted_eigenvalues, matrix):
    # calculating diagonal indices value in sigma matrix , and sort it from high to low  --->
    sigma_values = numpy.sqrt(sorted_eigenvalues)[::-1]
    sigma_values[numpy.isnan(sigma_values)] = 0  # replacing negative values with 0
    # Sigma is 243 *320
    sigma = numpy.zeros((len(matrix), len(matrix[0])), dtype=numpy.float64)
    # print(type(sorted_eigenvectors), len(sorted_eigenvectors), len(sorted_eigenvectors[0]), len(matrix), len(matrix[0]))

    # building sigma matrix
    for i in range(len(sigma)):
        sigma[i][i] = sigma_values[i]

    return sigma


def __build_U_matrix(matrix, sigma, V):
    # building U matrix
    # AV is 243 *320 , we need first 243 * 243 for U matrix.and they must be divided by sigma either
    AV = numpy.matmul(matrix, V)
    u_raw_version = AV[:, :243]
    u_raw_version_transpose = numpy.transpose(u_raw_version)
    for i in range(len(u_raw_version_transpose)):
        u_raw_version_transpose[i] /= sigma[i][i]
    U = numpy.transpose(u_raw_version_transpose)
    return U
