"""Module for data-related stuff."""

import numpy as np


def csv2numpy(file_name):
    """Read multidimensional signal from file"""
    # Read data from file.
    data = np.genfromtxt(file_name, delimiter=",", skip_header=1)
    # Return all columns but the first one (as it is the index).
    return data[:, 1:]
