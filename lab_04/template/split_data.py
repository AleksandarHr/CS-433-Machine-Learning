# -*- coding: utf-8 -*-
"""Exercise 3.

Split the dataset based on the given ratio.
"""


import numpy as np


def split_data(x, y, ratio, seed=1):
    """
    split the dataset based on the split ratio. If ratio is 0.8 
    you will have 80% of your data set dedicated to training 
    and the rest dedicated to testing
    """
    # set seed
    np.random.seed(seed)
    data_count = len(y)
    indices_permutation = np.random.permutation(data_count)
    cutoff_index = int(np.floor(ratio*data_count))
    x_test = x[[indices_permutation[cutoff_index:]]]
    y_test = y[[indices_permutation[cutoff_index:]]]
    x_train = x[[indices_permutation[:cutoff_index]]]
    y_train = y[[indices_permutation[:cutoff_index]]]
    return x_test,x_train,y_test,y_train