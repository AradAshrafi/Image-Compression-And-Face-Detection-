from utils.Utils import load_dataset_images, calculate_K_largest_eigs
import numpy
import cv2


def eigen_face():
    # load all 165 images from dataset,each has 243*320 size
    images = load_dataset_images()
    # calculate 243*320 average matrix
    avg_matrix = __calculate_avg_matrix(images_matrix=images)
    # calculate 165 * 243 * 320 subtracted (165 subtracted image for each matrix)
    subtracted_avg_matrix = __subtracted_avg_matrix(image_matrix=images, avg_matrix=avg_matrix)
    # calculate 243 * 243 covariance matrix
    covariance_matrix = __compute_covariance_matrix(subtracted_avg_matrix)
    # find 6 largest eigenvalues and eigenvectors of covariance matrix
    six_largest_eigen_values, six_largest_eigen_vectors = calculate_K_largest_eigs(square_matrix=covariance_matrix,
                                                                                   K=6)
    eigen_faces = __build_eigen_faces(largest_eigen_vectors=six_largest_eigen_vectors,
                                      subtracted_avg_matrix=subtracted_avg_matrix)
    for i in range(len(eigen_faces)):
        cv2.imwrite(str(i) + ".jpg", eigen_faces[i])


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


def __build_eigen_faces(largest_eigen_vectors, subtracted_avg_matrix):
    eigen_faces = numpy.zeros((6, 243, 320), dtype=numpy.float64)
    largest_eigen_vectors_transpose = numpy.transpose(largest_eigen_vectors)  # to iterate through them
    for i in range(len(largest_eigen_vectors_transpose)):  # range(6)
        for j in range(len(subtracted_avg_matrix)):  # range(165)
            eigen_faces[i] += largest_eigen_vectors_transpose[i][j] * subtracted_avg_matrix[j]
    return eigen_faces
