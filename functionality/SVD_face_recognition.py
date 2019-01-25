import numpy
from utils.Utils import load_dataset_images, calculate_avg_matrix, calculate_avg_subtraction
from utils.SVD import SVD_decomposition
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt


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
    U, sigma, V = SVD_decomposition(numpy.asarray(subtracted_train_image_matrices, dtype=numpy.float64))
    # best eigen images because they've being built with total rank of sigma
    X_array = numpy.matmul(numpy.transpose(U), subtracted_train_image_matrices)
    # # showing 10 best eigen images :-?
    # print("10 first eigen image : ")
    # for i in range(10):
    #     print(X_array[i])
    r_amounts = []
    r_order_approximation_error = []
    # rebuild subtracted sum with approximated value and new u,sigma,v
    for rank in range(15, 150, 25):
        r_amounts.append(rank)
        # calculate approximated U,sigma and V with rank =i
        u_prime, sigma_prime, v_prime = __calculate_r_order_svd_approximation(U, sigma, V,
                                                                              approximation_desired_rank=rank)
        approximated_sub = numpy.matmul(numpy.matmul(u_prime, sigma_prime), numpy.transpose(v_prime))
        r_order_approximation_error.append(mean_squared_error(subtracted_train_image_matrices, approximated_sub))

    __draw_r_order_approximation_error(r_amounts=r_amounts, r_order_approximation_error=r_order_approximation_error)

    r = 15  # it could be any thing between 0 and rank sigma
    Xr_array = __build_r_dimensional_feature_vector(U, subtracted_train_image_matrices, r)
    print()


def __convert_each_image_to_1D_array(images):
    return numpy.transpose(images.reshape(165, -1))


def __separate_train_and_test_images(images, training_ratio):
    # finding boundary column between test and train based on training ratio
    boundary_index = int(training_ratio * len(images[0]))
    train_images = images[:, :boundary_index]
    test_images = images[:, boundary_index + 1:]
    return train_images, test_images


# in approximation mode we need first r rows of V matrix
# in approximation mode we need first r columns of U matrix
# in approximation mode we just need first r rows and columns of this diagonal matrix
def __calculate_r_order_svd_approximation(U, sigma, V, approximation_desired_rank):
    V_prime = V[:, :approximation_desired_rank]
    U_prime = U[:, :approximation_desired_rank]
    sigma_prime = sigma[:approximation_desired_rank, :approximation_desired_rank]
    return U_prime, sigma_prime, V_prime


# draw approximation error for each r
def __draw_r_order_approximation_error(r_amounts, r_order_approximation_error):
    plt.plot(r_amounts, r_order_approximation_error)
    plt.show()


# building r dimension feature vector by multiplication of transpose of u in subtracted matrix
def __build_r_dimensional_feature_vector(u, subtracted_matrix, r):
    u_prime = u[:, :r]  # dividing our desired u from total u
    feature_vector = numpy.matmul(numpy.transpose(u_prime), subtracted_matrix)
    return feature_vector
