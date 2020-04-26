import numpy as np
import sys
from typing import Tuple


def roll_to_center(
    x: np.ndarray, y: np.ndarray, weights: np.ndarray, x0: float
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    if not (x.shape == y.shape == weights.shape) or not len(x.shape) == 1:
        raise ValueError("shape")
    if not np.all(np.diff(x) >= 0.0):
        raise ValueError("unsorted")
    if not (x[0] <= x0 < x[-1]):
        raise ValueError("center")

    # Find the center index.
    i0 = np.argmin(np.abs(x - x0))
    if x[i0] != x0:  # center is in between bins
        i0 += 1

    # Roll array so that center index is actually in the center.
    n2 = x.size // 2
    x = np.roll(x, n2 - i0)
    y = np.roll(y, n2 - i0)
    weights = np.roll(weights, n2 - i0)
    return x, y, weights


def symmetrize_profile(
    x: np.ndarray, y: np.ndarray, x0: float = 0.0, weights: np.ndarray = None
) -> Tuple[np.ndarray, np.ndarray]:
    if weights is None:
        weights = np.ones_like(y)

    x, y, weights = roll_to_center(x, y, weights, x0=x0)
    y *= weights
    y += np.flip(y)
    weights += np.flip(weights)

    n2 = x.size // 2
    x = x[n2:]
    y = y[n2:]
    y /= weights[n2:]

    return x, y


def antisymmetrize_profile(
    x: np.ndarray,
    y: np.ndarray,
    x0: float = 0.0,
    y0: float = 0.0,
    weights: np.ndarray = None,
) -> Tuple[np.ndarray, np.ndarray]:
    if weights is None:
        weights = np.ones_like(y)

    x, y, weights = roll_to_center(x, y, weights, x0=x0)
    y -= y0
    y *= weights
    y -= np.flip(y)
    weights += np.flip(weights)

    n2 = x.size // 2
    x = x[n2:]
    y = y[n2:]
    y /= weights[n2:]
    y += y0

    return x, y
