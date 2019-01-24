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
    for i in range(len(sorted_eigenvalues)):
        sigma[i][i] = sigma_values[i]

    return sigma


# to prevent memory error i make some changes in my U matrix,it wont return square matrix
# for example in the las phase it'll return 77760*150
def __build_U_matrix(matrix, sigma, V):
    # building U matrix
    # for instance,in eigen face section:
    # AV is 243 *320 , first we need to separate 243 * 243 for U matrix then they must be divided by sigma either
    AV = numpy.matmul(matrix, V)
    # extracting first len(Sigma) columns of AV, then we must divide each column by corresponding sigma value
    u_raw_version = AV[:, :len(sigma)]  # u is square matrix so both of it dimensions are len(sigma)
    # because math operations are row operations,for easier calculation i define u_transpose
    # at last i will again transpose it to form U matrix
    u_raw_version_transpose = numpy.transpose(u_raw_version)
    for i in range(len(u_raw_version_transpose)):
        if sigma[i][i] == 0:
            break
        u_raw_version_transpose[i] /= sigma[i][i]
    # transposing again to build U matrix
    # U = numpy.zeros((len(sigma), len(sigma)), dtype=float) <--------------- commented to prevent memory error
    U = numpy.transpose(u_raw_version_transpose)
    return U
