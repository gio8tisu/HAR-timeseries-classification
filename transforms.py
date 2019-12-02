"""Module for data transformation and normalization stuff."""

import numpy as np


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


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    x = np.linspace(0, 10, num=500)
    y = np.sin(x)
    res = stft(y.reshape((500, 1)), window_size=256)

    plt.plot(res[0])
    plt.show()
    plt.plot(res[1])
    plt.show()
