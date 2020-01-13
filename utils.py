from statistics import mode

import numpy as np
from sklearn.base import BaseEstimator

from dataset import HARDataset


def classify_time_series(ts, clf, window_length, metadata=None):
    """Return predicted class of time-series (ts).

    Predicts class for each window and return the
    most represented class.

    :type ts: np.ndarray
    :type clf: BaseEstimator
    :type window_length: int
    :type metadata: list or None.
    """
    serie = []
    for w in range(0, ts.shape[0], window_length):
        crop = ts[w:(w + window_length)]
        if metadata is not None:
            crop = np.concatenate([crop, metadata])
        if crop.shape[0] == (window_length if metadata is None
                             else window_length + len(metadata)):
            serie.append(clf.predict(crop.reshape((1, -1))).item())

    return mode(serie)


def make_sklearn_dataset(dataset):
    """Return features and class matrices from dataset.

    :type dataset: HARDataset

    TODO: define different "kinds" of features.
    """
    try:
        # Unpack metadata.
        X, y = zip(*[(np.hstack([np.linalg.norm(sample[:, -3:], axis=1)],
                                + metadata),
                      cls) for sample, cls, metadata in dataset])
    except ValueError:
        # No metadata in dataset.
        X, y = zip(*[(np.linalg.norm(sample[:, -3:], axis=1),
                      cls) for sample, cls in dataset])

    # Convert to NumPy ndarray and return.
    return np.array(X), np.array(y)
