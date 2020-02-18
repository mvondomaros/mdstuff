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


class PDens2D(Histogram2D):
    """A two-dimensional probability density."""

    def __init__(
        self,
        x_function: StructureFunction,
        x_bounds: Tuple[float, float],
        x_bin_width: float,
        y_function: StructureFunction,
        y_bounds: Tuple[float, float],
        y_bin_width: float,
        x_mode: str = "linear",
        y_mode: str = "linear",
    ) -> None:
        """
        :param x_function: the structure function for the x-channel
        :param x_bounds: the lower and upper bounds of the x-channel
        :param x_bin_width: the bin width of the x-channel
        :param y_function: the structure function for the y-channel
        :param y_bounds:  the lower and upper bounds of the y-channel
        :param y_bin_width: the bin width of the y-channel
        :param x_mode: optional, "linear" or "radial", the normalization scheme for the x-channel
        :param y_mode: optional, "linear" or "radial", the normalization scheme for the y-channel
        """
        super().__init__(
            x_function=x_function,
            x_bounds=x_bounds,
            x_bin_width=x_bin_width,
            y_function=y_function,
            y_bounds=y_bounds,
            y_bin_width=y_bin_width,
        )

        if x_mode not in ["linear", "radial"]:
            raise MDStuffError(f"invalid x_mode: {x_mode}")
        if y_mode not in ["linear", "radial"]:
            raise MDStuffError(f"invalid y_mode: {y_mode}")

        self.x_mode = x_mode
        self.y_mode = y_mode

    def get(self, centers: bool = False) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Return the probability densities and the bin edges.

        :parameter centers: optional, return the bin centers instead of the bin edges
        :return: the probability densities and the bin edges/centers
        """
        counts = self.counts.T.copy()
        n = np.sum(counts)
        if n > 0.0:
            counts /= n
        counts /= _normalizing_volume(self.x_bins, mode=self.x_mode)
        counts /= _normalizing_volume(self.y_bins, mode=self.y_mode)
        if centers:
            return counts, self.x_bins.centers.copy(), self.y_bins.centers.copy()
        else:
            return counts, self.x_bins.edges.copy(), self.y_bins.edges.copy()
