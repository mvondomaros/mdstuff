import numpy as np


def nextpow2(n: int) -> int:
    """
    Return the smallest power of 2 equal or greater than n.

    Args:
        n (int): An integer.

    Returns:
        int: The next power of 2.
    """
    return 1 << (n - 1).bit_length()


def autocorr_fft(x: np.ndarray, length: int = None) -> np.ndarray:
    """
    Compute the autocorrelation function of a one-dimensional array using FFT.

    Args:
        x (np.ndarray): The array.
        length (int, optional): The maximum correlation length. Defaults to None.

    Returns:
        np.ndarray: The autocorrelation function of the array.
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


def msd_fft(x: np.ndarray, length: int = None) -> np.ndarray:
    """
    Compute the mean squared displacements of a one-dimensional array.
    See: https://stackoverflow.com/questions/34222272/computing-mean-square-displacement-using-python-and-fft

    Args:
        x (np.ndarray): The positions.
        length (int, optional): The maximum correlation length. Defaults to None.

    Returns:
        np.ndarray: The mean squared displacements.
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
