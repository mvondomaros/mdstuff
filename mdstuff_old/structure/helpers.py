import numpy as np


def apply_mic(d: np.ndarray, box: float) -> None:
    """
    Apply the minimum image convention. Works in-place.

    :param d: distance vectors
    :param box: the box specification
    """
    box_inv = 1.0 / box
    np.subtract(d, box * np.rint(d * box_inv), out=d)
