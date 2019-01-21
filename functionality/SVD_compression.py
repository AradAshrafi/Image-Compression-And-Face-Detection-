from utils.Utils import load_image, SVD_decomposition
import sys
import numpy
import scipy.sparse


def SVD_compression():
    k = 100
    image = load_image()
    print(sys.getsizeof(image))
    U, sigma, V = SVD_decomposition(matrix=image)
    for i in range(k):
        sigma[242 - i][242 - i] = 0
    sigma = scipy.sparse.lil_matrix(sigma).todense()
    print(sys.getsizeof(numpy.matmul(numpy.matmul(U, sigma), V)))
