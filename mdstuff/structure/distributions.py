from typing import Tuple, Union

import numpy as np

from .base import Bins, Histogram, Histogram2D, StructureFunction
from .. import MDStuffError


def _normalizing_volume(bins: Bins, mode: str) -> Union[float, np.ndarray]:
    if mode == "linear":
        return bins.bin_width
    elif mode == "radial":
        return (1.0 / 3.0) * (bins.edges[1:] ** 3 - bins.edges[:-1] ** 3)
    else:
        raise NotImplementedError(f"{mode=}")


class PDens(Histogram):
    """A one-dimensional probability density."""

    def __init__(
        self,
        function: StructureFunction,
        bounds: Tuple[float, float],
        bin_width: float,
        mode: str = "linear",
    ) -> None:
        """
        :param function: the structure function
        :param bounds: the lower and upper bounds of the histogram
        :param bin_width: the bin width of the histogram
        :param mode: optional, the normalization mode ("linear" or "radial")
        """
        if mode not in ["linear", "radial"]:
            raise MDStuffError(f"invalid parameter value: {mode=}")

        super().__init__(function=function, bounds=bounds, bin_width=bin_width)

        self.mode = mode

    def get(self, centers: bool = False) -> Tuple[np.ndarray, np.ndarray]:
        """
        Return the probability densities and the bin edges.

        :parameter centers: optional, return the bin centers instead of the bin edges
        :return: the probability densities and the bin edges/centers
        """

        counts = self.counts.copy()
        n = np.sum(counts)
        if n > 0.0:
            counts /= n
        counts /= _normalizing_volume(bins=self.bins, mode=self.mode)
        if centers:
            return counts, self.bins.centers.copy()
        else:
            return counts, self.bins.edges.copy()
