from utils.Utils import load_images_from_folder
import numpy


def eigen_face():
    images = load_images_from_folder()
    avg_matrix = __calculate_avg_matrix(images_matrix=images)
    subtracted_avg_matrix = __subtracted_avg_matrix(image_matrix=images, avg_matrix=avg_matrix)



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

