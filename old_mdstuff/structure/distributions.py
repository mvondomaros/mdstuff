import numpy as np
import sys
from typing import Tuple, Union

from .base import Bins, Histogram, Histogram2D, StructureFunction
from .. import MDStuffError


def _normalizing_volume(bins: Bins, mode: str) -> Union[float, np.ndarray]:
    if mode == "linear":
        return bins.bin_width
    elif mode == "radial":
        return (1.0 / 3.0) * (bins.edges[1:] ** 3 - bins.edges[:-1] ** 3)
    elif mode == "angular":
        return np.cos(bins.edges[:-1] * np.pi / 180.0) - np.cos(
            bins.edges[1:] * np.pi / 180.0
        )
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
        :param mode: optional, the normalization mode ("linear", "radial", or "angular")
        """
        if mode not in ["linear", "radial", "angular"]:
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
        :param x_function: the structure function for the x-axis
        :param x_bounds: the lower and upper bounds for the x-axis
        :param x_bin_width: the bin width for the x-axis
        :param y_function: the structure function for the y-axis
        :param y_bounds:  the lower and upper bounds for the y-axis
        :param y_bin_width: the bin width for the y-axis
        :param x_mode: optional, "linear", "radial", or "angular": the normalization scheme for the x-axis
        :param y_mode: optional, "linear", "radial", or "angular": the normalization scheme for the y-axis
        """
        super().__init__(
            x_function=x_function,
            x_bounds=x_bounds,
            x_bin_width=x_bin_width,
            y_function=y_function,
            y_bounds=y_bounds,
            y_bin_width=y_bin_width,
        )

        if x_mode not in ["linear", "radial", "angular"]:
            raise MDStuffError(f"invalid x_mode: {x_mode}")
        if y_mode not in ["linear", "radial", "angular"]:
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
            counts /= np.atleast_2d(_normalizing_volume(self.x_bins, mode=self.x_mode))
            counts /= np.atleast_2d(
                _normalizing_volume(self.y_bins, mode=self.y_mode)
            ).T
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
        counts = self.counts.T.copy()
        hx = np.sum(counts, axis=0)
        hy = np.sum(counts, axis=1)
        hxy = np.tensordot(hx, hy, axes=0).T
        n = np.sum(counts)
        if n > 0.0:
            counts /= n
            counts /= np.atleast_2d(_normalizing_volume(self.x_bins, mode=self.x_mode))
            counts /= np.atleast_2d(
                _normalizing_volume(self.y_bins, mode=self.y_mode)
            ).T
        nxy = np.sum(hxy)
        if n > 0.0:
            hxy /= nxy
            hxy /= np.atleast_2d(_normalizing_volume(self.x_bins, mode=self.x_mode))
            hxy /= np.atleast_2d(_normalizing_volume(self.y_bins, mode=self.y_mode)).T
        counts -= hxy
        if centers:
            return counts, self.x_bins.centers.copy(), self.y_bins.centers.copy()
        else:
            return counts, self.x_bins.edges.copy(), self.y_bins.edges.copy()


class Profile(Histogram):
    """
    A one-dimensional profile: values of some property y accumulated in bins corresponding to a property x.
    Most often x is a distance/position.
    """

    def __init__(
        self,
        x_function: StructureFunction,
        x_bounds: Tuple[float, float],
        x_bin_width: float,
        y_function: StructureFunction,
    ):
        """
        :param x_function: the structure function for the x-axis
        :param x_bounds: the lower and upper bounds for the axis
        :param x_bin_width: the bin width for the x-axis
        :param y_function: the structure function for the y-axis
        """
        super().__init__(
            function=x_function, bounds=x_bounds, bin_width=x_bin_width,
        )
        self.x_function = x_function
        self.y_function = y_function
        self.nr_updates = 0

    def update(self) -> None:
        """
        """
        x = self.x_function()
        y = self.y_function()
        counts, _ = np.histogram(x, bins=self.bins.edges, weights=y)
        self.counts += counts
        self.nr_updates += 1

    def get(self, centers: bool = False) -> Tuple[np.ndarray, np.ndarray]:
        """
        """
        counts, bins = super().get(centers=centers)
        if self.nr_updates > 0:
            counts /= self.nr_updates
        return counts, bins


class AverageProfile(Histogram):
    """
    A one-dimensional profile: average values of some property y evaluated in bins corresponding to a property x.
    Most often x is a distance/position.
    """

    def __init__(
        self,
        x_function: StructureFunction,
        x_bounds: Tuple[float, float],
        x_bin_width: float,
        y_function: StructureFunction,
    ):
        """
        :param x_function: the structure function for the x-axis
        :param x_bounds: the lower and upper bounds for the axis
        :param x_bin_width: the bin width for the x-axis
        :param y_function: the structure function for the y-axis
        """
        super().__init__(
            function=x_function, bounds=x_bounds, bin_width=x_bin_width,
        )
        self.x_function = x_function
        self.y_function = y_function
        self.nr_updates = np.zeros_like(self.counts)

    def update(self) -> None:
        """
        """
        x = self.x_function()
        y = self.y_function()
        counts, _ = np.histogram(x, bins=self.bins.edges, weights=y)
        self.counts += counts

        counts, _ = np.histogram(x, bins=self.bins.edges)
        self.nr_updates += counts

    def get(self, centers: bool = False) -> Tuple[np.ndarray, np.ndarray]:
        """
        """
        counts, bins = super().get(centers=centers)
        np.true_divide(counts, self.nr_updates, out=counts, where=self.nr_updates > 0)
        return counts, bins


class DensityProfile(Profile):
    """
    A one-dimensional density-profile:values of some property y *per volume* accumulated in bins corresponding
    to a property x. Most often x is a distance/position.
    """

    def __init__(
        self,
        x_function: StructureFunction,
        x_bounds: Tuple[float, float],
        x_bin_width: float,
        y_function: StructureFunction,
        mode: str = "z",
    ) -> None:
        """
        :param x_function: the structure function for the x-axis
        :param x_bounds: the lower and upper bounds for the x-axis
        :param x_bin_width: the bin width for the x-axis
        :param y_function: the structure function for the y-axis
        :param mode: optional, "x", "y", or "z" the normalization scheme
        """
        super().__init__(
            x_function=x_function,
            x_bounds=x_bounds,
            x_bin_width=x_bin_width,
            y_function=y_function,
        )

        if mode not in "xyz":
            raise MDStuffError(f"invalid parameter argument: {mode=}")
        self.mode = mode

    def update(self) -> None:
        """
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

        x = self.x_function()
        y = self.y_function() / volume
        counts, _ = np.histogram(x, bins=self.bins.edges, weights=y)
        self.counts += counts
        self.nr_updates += 1
