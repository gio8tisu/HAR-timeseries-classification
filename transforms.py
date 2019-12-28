"""Module for data transformation and normalization stuff."""

import numpy as np
import scipy.signal
from sklearn.base import TransformerMixin, BaseEstimator


def stft(signal: np.ndarray, window_size=256, copy=True, pad=True):
    """Return the short time Fourier transform of signal.

    :param signal: numpy array of shape (L, N).
    :param window_size: window length.
    :param copy: whether to make a copy of signal.
    :param pad: whether to zero-pad input (if True) or truncate (if False).
    :return: numpy array of shape (M, window_size, N).

    Divide signal into chunks of length window_size and compute
    their Fourier transforms. If signal's length is not a multiple
    of window_size it will be filled with zeroes if pad is True
    and truncated otherwise.
    """
    assert isinstance(signal, np.ndarray)
    assert len(signal.shape) == 2

    if copy:
        signal = np.copy(signal)

    # Zero-pad or truncate if necessary.
    windows, remainder = divmod(signal.shape[0], window_size)
    if remainder != 0:
        if pad:
            padding = window_size * (windows + 1) - signal.shape[0]
            signal = np.pad(signal, ((0, padding), (0, 0)))
            windows += 1
        else:
            signal = signal[:(window_size * windows), :]

    # Reshape into "windowed view".
    signal = signal.reshape((windows, window_size, -1))

    # Compute FFT for each channel.
    signal_stft = np.empty_like(signal)
    for c in range(signal.shape[2]):
        signal_stft[..., c] = np.absolute(np.fft.fft(signal[..., c]))

    return signal_stft


def autocorrelation_peaks(signal, normalize=True):
    """

    :param signal: (L, d) numpy array.
    :param normalize: whether to normalize by value at lag-0 or not.
    """
    lag_0 = signal.shape[0] - 1
    autocorrelations = []
    for dim in range(signal.shape[1]):
        # Compute autocorrelation of dimension <dim>.
        corr = scipy.signal.correlate(signal[:, dim], signal[:, dim])
        # "Discard" first half.
        corr = corr[lag_0:]
        if normalize:
            # Divide by value at lag-0.
            corr = corr / corr[0]

        autocorrelations.append(corr)

    return autocorrelations


def frequency_peaks(signal):
    frequencies = []
    for dim in range(signal.shape[1]):
        # Compute fft of dimension <dim>.
        fft = np.fft.fft(signal[:, dim], signal[:, dim])
        # Only positive frequencies.
        fft = fft[:(signal.shape[0] / 2)]

        frequencies.append(np.abs(fft))

    return frequencies


class NormTransformer(BaseEstimator, TransformerMixin):

    def transform(self, X, y=None, **fit_params):
        return np.linalg.norm(X, axis=1)

    def fit(self, X, y=None, **fit_params):
        return self


class FourierTransform(BaseEstimator, TransformerMixin):

    def __init__(self, window=None):
        self.window = window

    def transform(self, X, y=None, **fit_params):
        # Compute fft of dimension <dim>. Only positive frequencies.
        if self.window is not None:
            X = X * self.window
        fft = np.fft.rfft(X)
        # Compute absolute value.
        return np.abs(fft)

    def fit(self, X, y=None, **fit_params):
        return self


class AutocorrelationTransform(BaseEstimator, TransformerMixin):

    def __init__(self, normalize):
        self.normalize = normalize

    def transform(self, X, y=None, **fit_params):
        corr = scipy.signal.correlate(X, X)
        # "Discard" first half.
        corr = corr[(X.shape[0] - 1):]
        if self.normalize:
            # Divide by value at lag_0.
            corr = corr / corr[0]
        return corr

    def fit(self, X, y=None, **fit_params):
        return self


class FindPeaksTransform(BaseEstimator, TransformerMixin):

    def __init__(self, n_peaks, height=0, **scipy_kwargs):
        self.n_peaks = n_peaks
        self.scipy_kwargs = scipy_kwargs
        self.height = height

    def transform(self, X, y=None, **fit_params):
        peaks, properties = scipy.signal.find_peaks(X, height=self.height,
                                                    **self.scipy_kwargs)
        # TODO sort peaks by their height.
        return peaks

    def fit(self, X, y=None, **fit_params):
        return self


class MeanTransform(BaseEstimator, TransformerMixin):

    def fit(self, X, y=None, **fit_params):
        return self

    def transform(self, X, y=None, **fit_params):
        return X.mean(axis=1, keepdims=True)


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    x = np.linspace(0, 10, num=500)
    y = np.sin(x)
    res = stft(y.reshape((500, 1)), window_size=256)

    plt.plot(res[0])
    plt.show()
    plt.plot(res[1])
    plt.show()
