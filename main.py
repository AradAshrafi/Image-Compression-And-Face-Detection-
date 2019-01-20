from functionality.eigen_face import eigen_face
from PIL import Image
import numpy

if __name__ == '__main__':
    eigen_face()
    a = numpy.asarray([[1, 2, 3], [2, 3, 4]])
    a += a
    print(a)
