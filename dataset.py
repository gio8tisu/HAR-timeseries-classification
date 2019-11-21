"""Module for data-related stuff."""

import numpy as np


def csv2numpy(root):
    data = np.genfromtxt(root, delimiter=",", skip_header=1)
    return data[:, 1:]
