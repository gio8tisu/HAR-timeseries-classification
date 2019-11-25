"""Module for data-related stuff."""

import pathlib

import numpy as np


def csv2numpy(file_name):
    """Read multidimensional signal from file"""
    # Read data from file.
    data = np.genfromtxt(file_name, delimiter=",", skip_header=1)
    # Return all columns but the first one (as it is the index).
    return data[:, 1:]


class HARDataset:

    CLASSES = {
        "dws": "down-stairs",
        "ups": "up-stairs",
        "wlk": "walk",
        "std": "standing",
        "sit": "sitting",
        "jog": "jogging"
    }

    FEATURES = [
        "attitude.roll",
        "attitude.pitch",
        "attitude.yaw",
        "gravity.x",
        "gravity.y",
        "gravity.z",
        "rotationRate.x",
        "rotationRate.y",
        "rotationRate.z",
        "userAcceleration.x",
        "userAcceleration.y",
        "userAcceleration.z"
    ]

    def __init__(self, data_root):
        self.data_root = pathlib.Path(data_root)
        self.files = []

        for csv in self.data_root.glob("**/*.csv"):
            class_ = str(csv.parent.stem)[:3]
            if class_ in self.CLASSES.keys():
                self.files.append([csv, class_])

    def __getitem__(self, item):
        file_, class_ = self.files[item]
        return csv2numpy(file_), class_

    def __len__(self):
        return len(self.files)
