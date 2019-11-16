# -*- coding: utf-8 -*-
"""Exercise 3.

Ridge Regression
"""

import numpy as np

def compute_mse(y, tx, w):
    error_vector = y - tx.dot(w)
    N = len(error_vector)
    mse = error_vector.ravel().dot(error_vector.ravel()) / (2 * N)
    return mse

def ridge_regression(y, tx, lambda_):
    """implement ridge regression."""
    lmbd = 2*tx.shape[0]*lambda_
    M = tx.T.dot(tx) + lmbd * np.identity(tx.shape[1])
    v = tx.T.dot(y)
    w = np.linalg.solve(M,v)
    return w, compute_mse(y, tx, w)

def rmse_ridge_regression(y, tx, lambda_):
    """implement ridge regression."""
    lmbd = 2*tx.shape[0]*lambda_
    M = tx.T.dot(tx) + lmbd * np.identity(tx.shape[1])
    v = tx.T.dot(y)
    w = np.linalg.solve(M,v)
    return w, compute_root_mse(y, tx, w)