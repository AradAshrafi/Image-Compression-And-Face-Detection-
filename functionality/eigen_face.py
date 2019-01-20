from utils.Utils import load_images_from_folder, calculate_K_largest_eigs
import numpy


def eigen_face():
    images = load_images_from_folder()
    avg_matrix = __calculate_avg_matrix(images_matrix=images)
    subtracted_avg_matrix = __subtracted_avg_matrix(image_matrix=images, avg_matrix=avg_matrix)
    covariance_matrix = __compute_covariance_matrix(subtracted_avg_matrix)
    six_largest_eigen_values, six_largest_eigen_vectors = calculate_K_largest_eigs(symmetric_matrix=covariance_matrix,
                                                                                   K=6)

    print(six_largest_eigen_values, six_largest_eigen_vectors)


def __calculate_avg_matrix(images_matrix):
    avg_matrix = numpy.asarray(images_matrix[0].copy())
    for i in range(len(images_matrix) - 1):
        avg_matrix += images_matrix[i + 1]
    avg_matrix = avg_matrix / len(images_matrix)
    return avg_matrix


def __subtracted_avg_matrix(image_matrix, avg_matrix):
    subtracted_avg_matrix = image_matrix.copy()
    subtracted_avg_matrix -= avg_matrix
    return subtracted_avg_matrix


def __compute_covariance_matrix(subtracted_avg_matrix):
    covariance_matrix = numpy.zeros((243, 243), dtype=numpy.float64)
    for image in subtracted_avg_matrix:
        image_transpose = numpy.transpose(image) / len(subtracted_avg_matrix)
        covariance_matrix += numpy.matmul(image, image_transpose)

    return covariance_matrix
