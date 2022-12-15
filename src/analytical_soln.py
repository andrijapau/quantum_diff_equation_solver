import numpy as np
import scipy as sp


def solve_p(t, M, p0, b):
    '''
    Finds the analytic solution for the matrix LDE: dp/dt = M * p0 + b.
    '''
    p = np.matmul(sp.linalg.expm(M * t), p0) + np.matmul((sp.linalg.expm(M * t) - np.identity(2)),
                                                         np.matmul(np.linalg.inv(M), b))
    return p
