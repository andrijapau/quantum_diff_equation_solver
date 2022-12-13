import numpy as np
import scipy as sp


def solve_p(t, p0, b):
    sigma_x = np.array([[0, 1], [1, 0]])
    I = np.array([[1, 0], [0, 1]])

    p = np.matmul(sp.linalg.expm(sigma_x * t), p0) + np.matmul((sp.linalg.expm(sigma_x * t) - I), np.matmul(sigma_x, b))
    # print(p.shape)
    return p
