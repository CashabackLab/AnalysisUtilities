from numba import njit
import numba as nb
import numpy as np

@nb.jit
def quadratic_func(data_x, data_y):
    # https://math.stackexchange.com/questions/1687737/least-squares-fitting-quadratic-equation-to-a-set-of-points
    n = len(data_x)
    x_sum = np.sum(data_x)
    x_squared_sum = np.sum(data_x**2)
    x_cubed_sum = np.sum(data_x**3)
    x_quad_sum = np.sum(data_x**4)
    y_sum = np.sum(data_y)
    xy_sum = np.sum(data_x*data_y)
    xsquaredy_sum = np.sum((data_x**2)*data_y)

    A_matrix = np.array([[n, x_sum, x_squared_sum],
                [x_sum, x_squared_sum, x_cubed_sum],
                [x_squared_sum, x_cubed_sum, x_quad_sum]])

    B_matrix = np.array([[y_sum], [xy_sum], [xsquaredy_sum]])
    
    c, b, a = np.linalg.solve(A_matrix, B_matrix)
    return a[0], b[0], c[0]