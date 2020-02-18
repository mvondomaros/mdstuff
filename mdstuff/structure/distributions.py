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


class CorrFunc2D(PDens2D):
    """A two-dimensional correlation function."""

    def get(self, centers: bool = False) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Return the probability densities and the bin edges.

        :parameter centers: optional, return the bin centers instead of the bin edges
        :return: the probability densities and the bin edges/centers
        """
        counts, _, _ = super().get()
        hx = np.sum(counts, axis=0)
        nx = np.sum(hx)
        if nx > 0.0:
            hx /= nx
            hx /= _normalizing_volume(bins=self.x_bins, mode=self.x_mode)
        hy = np.sum(counts, axis=1)
        ny = np.sum(hy)
        if ny > 0.0:
            hy /= ny
            hy /= _normalizing_volume(bins=self.y_bins, mode=self.y_mode)
        hxy = np.tensordot(hx, hy, axes=0).T
        nxy = np.sum(hxy)
        if nxy > 0.0:
            hxy /= nxy
            hxy /= _normalizing_volume(bins=self.x_bins, mode=self.x_mode)
            hxy /= _normalizing_volume(bins=self.y_bins, mode=self.y_mode)
        counts -= hxy
        if centers:
            return counts, self.x_bins.centers.copy(), self.y_bins.centers.copy()
        else:
            return counts, self.x_bins.edges.copy(), self.y_bins.edges.copy()


class Prof(Histogram):
    """A one-dimensional profile."""

    def __init__(
        self,
        function: StructureFunction,
        bounds: Tuple[float, float],
        bin_width: float,
        weight_function: StructureFunction,
    ) -> None:
        """
        :param function: the structure function that defines the histogram
        :param bounds: the lower and upper bounds of the histogram
        :param bin_width: the bin width of the histogram
        :param weight_function: the structure function that computes the weights
        """
        super().__init__(
            function=function,
            bounds=bounds,
            bin_width=bin_width,
            weight_function=weight_function,
        )
        self.nr_updates = 0

    def update(self) -> None:
        """
        Add weighted values to the histogram.
        """
        super().update()
        self.nr_updates += 1

    def get(self, centers: bool = False) -> Tuple[np.ndarray, np.ndarray]:
        """
        Return the probability densities and the bins edges.

        :parameter centers: optional, return the bin centers instead of the bin edges
        :return: the probability densities and the bin edges/centers
        """
        counts, bins = super().get(centers=centers)
        if self.nr_updates > 0:
            counts /= self.nr_updates
        return counts, bins


class DProf(Prof):
    """A one-dimensional density-profile."""

    def __init__(
        self,
        function: StructureFunction,
        bounds: Tuple[float, float],
        bin_width: float,
        weight_function: StructureFunction,
        mode: str = "z",
    ) -> None:
        """
        :param function: the structure function that defines the histogram
        :param bounds: the lower and upper bounds of the histogram
        :param bin_width: the bin width of the histogram
        :param weight_function: the structure function that computes the weights
        :param mode: optional, "x", "y", or "z" the normalization scheme
        """
        super().__init__(
            function=function,
            bounds=bounds,
            bin_width=bin_width,
            weight_function=weight_function,
        )

        if mode not in "xyz":
            raise MDStuffError(f"invalid parameter argument: {mode=}")
        self.mode = mode

    def update(self) -> None:
        """
        Add weighted values to the histogram.
        """
        if self.mode == "x":
            volume = (
                self.universe.dimensions[1]
                * self.universe.dimensions[2]
                * self.bins.bin_width
            )
        elif self.mode == "y":
            volume = (
                self.universe.dimensions[0]
                * self.universe.dimensions[2]
                * self.bins.bin_width
            )
        elif self.mode == "z":
            volume = (
                self.universe.dimensions[0]
                * self.universe.dimensions[1]
                * self.bins.bin_width
            )
        else:
            raise NotImplementedError(f"{self.mode=}")

        values = self.function()
        weights = self.weight_function() / volume
        counts, _ = np.histogram(values, bins=self.bins.edges, weights=weights)
        self.counts += counts
        self.nr_updates += 1
