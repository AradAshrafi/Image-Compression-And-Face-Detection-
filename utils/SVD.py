import numpy
from utils.Utils import calculate_eigs


def SVD_decomposition(matrix):
    AtA = numpy.matmul(matrix.transpose(), matrix)
    # finding eigenvalues and eigenvectors of matrix
    sorted_eigenvalues, sorted_eigenvectors = calculate_eigs(square_matrix=AtA)
    # Build sigma matrix from AtA eigenvalues
    sigma = __build_sigma_matrix(sorted_eigenvalues=sorted_eigenvalues, matrix=matrix)
    # building V matrix
    # V is 320 * 320
    V = numpy.transpose(numpy.transpose(sorted_eigenvectors)[::-1])

    # building U matrix
    U = __build_U_matrix(matrix=matrix, sigma=sigma, V=V)
    return U, sigma, V


def __build_sigma_matrix(sorted_eigenvalues, matrix):
    # calculating diagonal indices value in sigma matrix , and sort it from high to low  --->
    sigma_values = numpy.sqrt(sorted_eigenvalues)[::-1]
    sigma_values[numpy.isnan(sigma_values)] = 0  # replacing negative values with 0
    # Sigma is 243 *320
    sigma = numpy.zeros((len(matrix), len(matrix[0])), dtype=numpy.float64)
    # building sigma matrix
    for i in range(len(sigma)):
        sigma[i][i] = sigma_values[i]

    return sigma


def __build_U_matrix(matrix, sigma, V):
    # building U matrix
    # AV is 243 *320 , we need first 243 * 243 for U matrix.and they must be divided by sigma either
    AV = numpy.matmul(matrix, V)
    u_raw_version = AV[:, :243]
    u_raw_version_transpose = numpy.transpose(u_raw_version)
    for i in range(len(u_raw_version_transpose)):
        u_raw_version_transpose[i] /= sigma[i][i]
    U = numpy.transpose(u_raw_version_transpose)
    return U
