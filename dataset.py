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
    """Dataset with fixed-length crops.

    Args:
        data_root -- string. Path to data directory.
        length -- int. Crops length.
        discard_start -- int. Number of samples to discard from start.
        discard_end -- int. Number of samples to discard from end.
        unwrapped_attitude -- bool. Whether to unwrap attitude signals.
        padding_mode -- None or string. If None, the samples not fitting in
                integer number of windows will be discarded. If string,
                the value will be passed to numpy's pad function.
    """

    def __init__(self, data_root, length, discard_start, discard_end,
                 unwrapped_attitude=True, padding_mode=None):
        super().__init__(data_root, unwrapped_attitude=unwrapped_attitude)
        self.length = length
        self.discard_start = discard_start
        self.discard_end = discard_end
        self.padding_mode = padding_mode

        self.crops = self.get_crops()

    def get_crops(self):
        """Return list with crops from files."""
        crops = []
        # Iterate over data files.
        for file, class_ in self.files:
            # Read from file.
            signal = csv2numpy(file)
            # Crop start and end.
            signal = signal[self.discard_start:(signal.shape[0] - self.discard_end)]
            windows, remainder = divmod(signal.shape[0], self.length)
            if self.padding_mode and remainder != 0:
                # Apply padding with given padding mode.
                padding = self.length * (windows + 1) - signal.shape[0]
                signal = np.pad(signal, ((0, padding), (0, 0)), self.padding_mode)
            elif self.padding_mode is None:
                # Crop the end.
                signal = signal[:(self.length * windows)]
            # Obtain crops from <discard_start> to <discard-end>.
            for i in range(0, signal.shape[0], self.length):
                crop = signal[i:(i + self.length)]
                if self.unwrapped_attitude:
                    # Unwrap phase of first 3 features (attitude signals).
                    for s in range(3):
                        crop[:, s] = np.unwrap(crop[:, s])
                crops.append([crop, class_])

        return crops

    def __getitem__(self, item):
        return self.crops[item]

    def __len__(self):
        return len(self.crops)


if __name__ == '__main__':
    dataset = HARDatasetCrops('motionsense-dataset', 256, 10, 10, True)
    for item in iter(dataset):
        assert item[0].shape == (256, 12)
