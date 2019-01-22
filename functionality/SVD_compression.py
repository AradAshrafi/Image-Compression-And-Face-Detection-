from utils.Utils import load_image, draw_compression_plot
from utils.SVD import SVD_decomposition
import sys
import numpy
import scipy.sparse
from PIL import Image
import cv2


# compress image using SVD decomposition
def SVD_compression():
    image = load_image()
    orginal_matrix_size = __calculate_matrix_elements_summation(matrix=image)
    k_values = []
    compression_rates = []
    U, sigma, V = SVD_decomposition(matrix=image)
    # change singular values to zero
    for k in range(5, 125, 25):
        k_values.append(k)
        # clone sigma matrix,because in each iteration we change different diagonal values in it
        compressed_sigma = sigma.copy()
        for i in range(len(sigma) - k):
            compressed_sigma[k + i][k + i] = 0
        compressed_image = numpy.asarray(numpy.matmul(numpy.matmul(U, compressed_sigma), numpy.transpose(V)),
                                         dtype=numpy.uint8)
        current_compression_rate = 1 - __calculate_matrix_elements_summation(compressed_image) / orginal_matrix_size
        compression_rates.append(current_compression_rate)

    draw_compression_plot(k_values=k_values, compression_rates=compression_rates)


