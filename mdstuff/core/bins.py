from __future__ import annotations

import warnings
from typing import Tuple

import numpy as np

from .errors import MDStuffError


class Bins:
    """One-dimensional bins."""

    def __init__(self, bounds: Tuple[float, float], bin_width: float) -> None:
        """
        :param bounds: a tuple of lower and upper bounds
        :param bin_width: the bin width
        """
        if len(bounds) != 2 or np.isclose(bounds[0], bounds[1]):
            raise MDStuffError(f"invalid bounds specification: {bounds=}")
        if bin_width <= 0.0:
            raise MDStuffError(f"invalid bin width: {bin_width=}")

        self.bin_width = bin_width
        self.bounds = np.array(sorted(bounds))

        self.edges = np.arange(bounds[0], bounds[1] + bin_width, bin_width)
        if not np.isclose(self.edges[-1], bounds[1]):
            warnings.warn(
                f"bounds/bin width mismatch; new bounds are {(self.edges[0], self.edges[-1])}"
            )
        self.centers = 0.5 * (self.edges[1:] + self.edges[:-1])
        self.nr = self.centers.size
