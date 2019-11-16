# -*- coding: utf-8 -*-
"""A function to compute the cost."""

def compute_mse(y, tx, w):
    """compute the loss by mse."""
    e = y - tx.dot(w)
    mse = e.dot(e) / (2 * len(e))
    return mse

def compute_root_mse(y, tx, w):
    mse = compute_mse(y, tx, w)
    return math.sqrt(2 * mse)

