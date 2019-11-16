# -*- coding: utf-8 -*-
"""Exercise 3.

Least Square
"""

import numpy as np

def least_squares(y, tx):
    """calculate the least squares solution."""
    # Construct the normal equations
    M = tx.T.dot(tx)
    v = tx.T.dot(y)
    w = np.linalg.solve(M,v)
    loss = compute_mse(y, tx, w)
    return w, loss

def rmse_least_squares(y, tx):
    """calculate the least squares solution with rmse."""
    # Construct the normal equations
    M = tx.T.dot(tx)
    v = tx.T.dot(y)
    w = np.linalg.solve(M,v)
    return w, compute_root_mse(y, tx, w)