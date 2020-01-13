from statistics import mode
import abc

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin

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
    for i, w in enumerate(range(0, ts.shape[0], window_length)):
        if (i + 1) * window_length > ts.shape[0]:
            break
        crop_ = ts[w:(w + window_length)]
        crop = crop_.T.reshape((1, -1))
        if metadata is not None:
            crop = np.hstack([crop, metadata])
        serie.append(clf.predict(crop.reshape((1, -1))).item())

    return mode(serie)


class TimeSeriesClassifier(metaclass=abc.ABCMeta):
    def __init__(self, classifier, window_length):
        super().__init__()
        self.window_length = window_length
        self.classifier = classifier

    def predict(self, dataset):
        y_pred = []
        for x, _, meta in dataset:
            x = np.hstack([x[:, :-6], np.linalg.norm(x[:, -6:-3], axis=1, keepdims=True),
                           np.linalg.norm(x[:, -3:], axis=1, keepdims=True)])
            y_pred.append(classify_time_series(x, self.classifier, 256, [meta]))
        return y_pred

    @abc.abstractmethod
    def classify_time_series(self, ts, clf, window_length, metadata=None):
        pass


class ModeTimeSeriesClassifier(TimeSeriesClassifier):
    def classify_time_series(self, ts, clf, window_length, metadata=None):
        serie = []
        for i, w in enumerate(range(0, ts.shape[0], window_length)):
            if (i + 1) * window_length > ts.shape[0]:
                break
            crop_ = ts[w:(w + window_length)]
            crop = crop_.T.reshape((1, -1))
            if metadata is not None:
                crop = np.hstack([crop, metadata])
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
