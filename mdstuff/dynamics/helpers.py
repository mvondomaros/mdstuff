import numpy as np


def nextpow2(n: int) -> int:
    """
    Return the smallest power of 2 equal or greater than n.

    :param n: an integer
    :return: the next power of 2
    """
    return 1 << (n - 1).bit_length()


def autocorr_fft(x: np.ndarray, length: int = None):
    """
    Compute the autocorrelation function of a one-dimensional array using FFT.

    :param x: the array
    :param length: optional, the maximum correlation length
    :return: the autocorrelation function of the array
    """
    n = len(x)
    m = n if length is None else length
    # fourier transform
    res = np.fft.fft(x, n=nextpow2(n + m))
    # spectral power density
    res = res * res.conjugate()
    # inverse fourier transform
    res = np.fft.ifft(res)
    # shift
    res = (res[:m]).real
    return res


def msd_fft(x: np.ndarray, length: int = None):
    """
    Compute the mean squared displacements of a one-dimensional array.
    See: https://stackoverflow.com/questions/34222272/computing-mean-square-displacement-using-python-and-fft

    :param x: the array
    :param length: optional, the maximum correlation length
    :return: the mean squared displacements
    """
    n = x.size
    m = n if length is None else length
    norm = np.arange(n, n - m, -1)

    s2 = autocorr_fft(x, length=length)

    xsq = np.zeros(n + 1)
    xsq[:n] = x * x
    sum_xsq = 2.0 * np.sum(xsq)

    s1 = np.zeros(m)
    for i in range(m):
        sum_xsq = sum_xsq - xsq[i - 1] - xsq[n - i]
        s1[i] = sum_xsq

    return (s1 - 2.0 * s2) / norm
