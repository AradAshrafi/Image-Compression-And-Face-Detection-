from utils.Utils import load_image
from utils.SVD import SVD_decomposition
import sys
import numpy
import scipy.sparse
from PIL import Image
import cv2


def SVD_compression():
    k = 220
    image = load_image()
    print(sys.getsizeof(image))
    U, sigma, V = SVD_decomposition(matrix=image)
    for i in range(k):
        sigma[242 - i][242 - i] = 0

    compressed_image = numpy.matmul(numpy.matmul(U, sigma), numpy.transpose(V))
    # cv2.imwrite("eigenface outputs/testt.jpg", compressed_image)
    # result = Image.fromarray(compressed_image)
    # result.save('out.jpg')
    # sigma = scipy.sparse.lil_matrix(sigma).todense()
    # print(sys.getsizeof(numpy.matmul(numpy.matmul(U, sigma), V)))
