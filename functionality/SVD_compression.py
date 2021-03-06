from utils.Utils import load_image, draw_compression_plot, calculate_matrix_elements_summation, \
    convert_k_diagonal_elements_to_zero
from utils.SVD import SVD_decomposition
import numpy


# compress image using SVD decomposition
def SVD_compression():
    image = load_image()
    original_matrix_size = calculate_matrix_elements_summation(matrix=image)
    U, sigma, V = SVD_decomposition(matrix=image)
    k_values = []
    compression_rates = []
    # change singular values to zero
    # calculate compression rate for  k =5,30,55,80,105
    for k in range(5, 105, 25):
        k_values.append(k)
        # change diagonal values to zero from kth row
        compressed_sigma = convert_k_diagonal_elements_to_zero(start_index=k, matrix=sigma)
        # build compressed image with new sigma values
        compressed_image = numpy.asarray(numpy.matmul(numpy.matmul(U, compressed_sigma), numpy.transpose(V)),
                                         dtype=numpy.uint8)
        # calculate compression rate as following -> 1 - new_image_size/original_matrix_size
        current_compression_rate = 1 - calculate_matrix_elements_summation(compressed_image) / original_matrix_size
        compression_rates.append(current_compression_rate)

    # plotting compression rates for each k value
    draw_compression_plot(k_values=k_values, compression_rates=compression_rates)
