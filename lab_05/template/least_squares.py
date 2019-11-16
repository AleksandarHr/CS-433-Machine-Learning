# -*- coding: utf-8 -*-
"""Exercise 3.

Least Square
"""

import numpy as np

def compute_mse(y, tx, w):
    error_vector = y - tx.dot(w)
    N = y.shape[0]
    mse = error_vector.ravel().dot(error_vector.ravel()) / (2 * N)
    return mse

def least_squares(y, tx):
    """Compute the Least Squares Solution"""
    M = tx.T.dot(tx)
    v = tx.T.dot(y)
    w = np.linalg.solve(M,v)
    return w, compute_mse(y, tx, w)

