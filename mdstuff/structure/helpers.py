import numpy as np


def apply_mic_1d(x: np.ndarray, box: float) -> None:
    """
    Apply one-dimensional minimum image convention.

    :param x: a numpy array with distances
    :param box: the box length
    """
    box_inv = 1.0 / box
    # Make this work in-place.
    np.subtract(x, box * np.rint(x * box_inv), out=x)


def upper_triangular_view(x: np.ndarray):
    """Return a numpy view on the upper triangular part of a matrix. No type checking."""
    iu = np.triu_indices_from(x)
    return x[iu]
