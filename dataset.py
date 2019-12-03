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

    def __init__(self, data_root, unwrapped_attitude=False):
        self.data_root = pathlib.Path(data_root)
        self.files = []
        self.unwrapped_attitude = unwrapped_attitude

        for csv in self.data_root.glob("**/*.csv"):
            class_ = str(csv.parent.stem)[:3]
            if class_ in self.CLASSES.keys():
                self.files.append([csv, class_])

    def __getitem__(self, item):
        file_, class_ = self.files[item]
        if not self.unwrapped_attitude:
            return csv2numpy(file_), class_

        # Unwrap attitude signals.
        signals = csv2numpy(file_)
        for i in range(3):
            signals[:, i] = np.unwrap(signals[:, i])

        return signals, class_

    def __len__(self):
        return len(self.files)


class HARDatasetCrops(HARDataset):

    def __init__(self, data_root, length, discard_start, discard_end,
                 unwrapped_attitude=False):
        super().__init__(data_root, unwrapped_attitude=unwrapped_attitude)
        self.length = length
        self.discard_start = discard_start
        self.discard_end = discard_end

        self.crops = self.get_crops()

    def get_crops(self):
        """Return list with crops from files.

        TODO: zero-pad.
        """
        crops = []
        # Iterate over data files and obtain crops.
        for file, class_ in self.files:
            signal = csv2numpy(file)
            for i in range(self.discard_start,
                           signal.shape[0] - self.discard_end, self.length):
                crop = signal[i:(i + self.length)]
                if self.unwrapped_attitude:
                    for s in range(3):
                        crop[:, s] = np.unwrap(crop[:, s])
                crops.append([crop, class_])

        return crops

    def __getitem__(self, item):
        return self.crops[item]

    def __len__(self):
        return len(self.crops)
