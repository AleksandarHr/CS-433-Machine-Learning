# -*- coding: utf-8 -*-
"""implement a polynomial basis function."""

import numpy as np


def build_poly(x, degree):
    """polynomial basis functions for input data x, for j=0 up to j=degree."""
    poly_matrix = np.ones((len(x), 1))
    for pow in range (1, degree+1):
        poly_matrix = np.c_[poly_matrix, np.power(x, pow)]
    return poly_matrix
