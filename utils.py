from statistics import mode
import abc

import numpy as np
from sklearn.base import BaseEstimator

from dataset import HARDataset


class TimeSeriesClassifier(metaclass=abc.ABCMeta):
    """

    :type classifier: BaseEstimator
    :type window_length: int
    """

    def __init__(self, classifier, window_length, overlap=0):
        self.window_length = window_length
        self.classifier = classifier
        self.overlap = overlap

    def predict(self, dataset):
        """Return predicted classes for dataset time-series.

        :type dataset: HARDataset.
        """
        y_pred = []
        for sample in dataset:
            if len(sample) == 3:
                x, _, meta = sample
            else:
                x, _ = sample
                meta = None
            x = np.hstack([x[:, :-6],
                           np.linalg.norm(x[:, -6:-3], axis=1, keepdims=True),
                           np.linalg.norm(x[:, -3:], axis=1, keepdims=True)])
            if meta:
                y_pred.append(self.classify_time_series(x, [meta]))
            else:
                y_pred.append(self.classify_time_series(x))
        return np.array(y_pred)

    @abc.abstractmethod
    def classify_time_series(self, ts, metadata=None):
        """Return predicted class of time-series (ts).

        :type ts: np.ndarray
        :type metadata: list or None.
        """
        pass


class ModeTimeSeriesClassifier(TimeSeriesClassifier):

    def classify_time_series(self, ts, metadata=None):
        """Predicts class for each window and return the most represented class.
        """
        predictions = []
        w = 0
        while w + self.window_length < ts.shape[0]:
            crop_ = ts[w:(w + self.window_length)]
            crop = crop_.T.reshape((1, -1))
            if metadata is not None:
                crop = np.hstack([crop, metadata])
            predictions.append(
                self.classifier.predict(crop.reshape((1, -1))).item())
            w += self.window_length - self.overlap

        return mode(predictions)


class SumTimeSeriesClassifier(TimeSeriesClassifier):

    def classify_time_series(self, ts, metadata=None):
        """Predicts class for each window and return the most represented class.
        """
        predictions = []
        w = 0
        while w + self.window_length < ts.shape[0]:
            crop_ = ts[w:(w + self.window_length)]
            crop = crop_.T.reshape((1, -1))
            if metadata is not None:
                crop = np.hstack([crop, metadata])
            predictions.append(
                self.classifier.predict_proba(crop.reshape((1, -1))).item())
            w += self.window_length - self.overlap

        return np.array(predictions).sum(axis=0).argmax()


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
