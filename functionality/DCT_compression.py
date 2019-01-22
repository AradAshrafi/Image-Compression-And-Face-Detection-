from scipy.fftpack import dct, idct
from utils.Utils import load_image, convert_k_diagonal_elements_to_zero, calculate_matrix_elements_summation, \
    draw_compression_plot
import numpy


def DCT_compression():
    image = load_image()
    original_matrix_size = calculate_matrix_elements_summation(matrix=image)
    # convert to dct
    dct_converted_image = dct(image, norm="ortho")
    # will append k and compression rate for each k later
    k_values = []
    compression_rates = []
    # change singular values to zero
    # calculate compression rate for  k =5,30,55,80,105
    for k in range(5, 105, 25):
        k_values.append(k)
        # change diagonal values to zero
        compressed_dct_converted_image = convert_k_diagonal_elements_to_zero(start_index=k, matrix=dct_converted_image)
        # convert back to image form
        compressed_image = numpy.asarray(idct(compressed_dct_converted_image, norm="ortho"), dtype=numpy.uint8)
        # calculate compression rate as following -> 1 - new_image_size/original_matrix_size
        current_compression_rate = 1 - calculate_matrix_elements_summation(compressed_image) / original_matrix_size
        compression_rates.append(current_compression_rate)

    # plotting compression rates for each k value
    draw_compression_plot(k_values=k_values, compression_rates=compression_rates)
