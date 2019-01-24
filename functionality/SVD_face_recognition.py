import numpy
from utils.Utils import load_dataset_images, calculate_avg_matrix, calculate_avg_subtraction
from utils.SVD import SVD_decomposition


def SVD_face_recognition():
    training_ratio = 0.91
    # first load total dataset images
    images = load_dataset_images()
    # convert each 2D image to 1D
    images = __convert_each_image_to_1D_array(images=images)
    # separate train and test images
    train_images, test_images = __separate_train_and_test_images(images=images, training_ratio=training_ratio)
    # calculate avg of train faces
    # transpose train images to have each train vector in one row,because my avg calculator is based on rows
    train_faces_avg = numpy.transpose(calculate_avg_matrix(images_matrix=numpy.transpose(train_images)))
    # calculate subtracted matrix
    # same as above,because matrix summation and subtraction works easier on row vectors
    # first transpose them to build row avg and row image subtraction,then at last transpose it again
    subtracted_train_image_matrices = numpy.transpose(
        calculate_avg_subtraction(image_matrix=numpy.transpose(train_images),
                                  avg_matrix=numpy.transpose(train_faces_avg)))
    # perform SVD on this subtracted matrix
    U, sigma, V = SVD_decomposition(numpy.asarray(subtracted_train_image_matrices,dtype=numpy.float64))

    print(U[0:10])


def __convert_each_image_to_1D_array(images):
    return numpy.transpose(images.reshape(165, -1))


def __separate_train_and_test_images(images, training_ratio):
    # finding boundary column between test and train based on training ratio
    boundary_index = int(training_ratio * len(images[0]))
    train_images = images[:, :boundary_index]
    test_images = images[:, boundary_index + 1:]
    return train_images, test_images

