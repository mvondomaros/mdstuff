from __future__ import annotations

import abc
import warnings
from typing import Tuple

import numpy as np

from ..core import Analysis, MDStuffError, ParameterValueError, Universe


class Bins:
    """One-dimensional bins."""

    def __init__(self, bounds: Tuple[float, float], bin_width: float):
        """
        :param bounds: the lower and upper bounds
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


class StructureFunction(abc.ABC):
    """
    Base class for functions that work on a Universe.

    Function calls return a numpy array. The shape of the array can be inquired by the shape() property.
    """

    def __init__(self, universe: Universe):
        self.universe = universe
        self._shape = None

    @abc.abstractmethod
    def __call__(self, *args, **kwargs) -> np.ndarray:
        pass

    @property
    def shape(self) -> Tuple:
        return self._shape


class VectorReduction(StructureFunction, abc.ABC):
    def __init__(self, function: StructureFunction):
        if len(function.shape) != 2:
            raise MDStuffError(f"function does not return vectors")
        super().__init__(universe=function.universe)
        self.function = function

        self._shape = (function.shape[0],)

    @abc.abstractmethod
    def __call__(self) -> np.ndarray:
        pass


class Magnitude(VectorReduction):
    def __call__(self) -> np.ndarray:
        values = self.function()
        return np.linalg.norm(values, axis=1)


class Projection(VectorReduction):
    def __init__(self, function: StructureFunction, n: int = 2):
        super().__init__(function=function)

        if n not in [0, 1, 2]:
            raise MDStuffError(f"invalid parameter argument: {n=}")
        self.n = n

    def __call__(self) -> np.ndarray:
        values = self.function()
        return values[:, self.n]


class Orientation(Projection):
    def __call__(self) -> np.ndarray:
        values = self.function()
        magnitudes = np.linalg.norm(values, axis=1)
        return values[:, self.n] / magnitudes


class Histogram(Analysis):
    """A one-dimensional, running histogram."""

    def __init__(
        self,
        function: StructureFunction,
        bounds: Tuple[float, float],
        bin_width: float,
        weight_function: StructureFunction = None,
    ):
        """
        :param function: a structure function
        :param bounds: the lower and upper bounds of the histogram
        :param bin_width: the bin width of the histogram
        :param weight_function: optional, a structure function returning weights
        """
        super().__init__(universe=function.universe)

        if (
            weight_function is not None
            and weight_function.universe != function.universe
        ):
            raise MDStuffError(
                f"function and weight function work with different universes"
            )
        if len(function.shape) != 1:
            raise MDStuffError(f"function does not return a one-dimensional array")
        if weight_function is not None and len(weight_function.shape) != 1:
            raise MDStuffError(
                f"weight function does not return a one-dimensional array"
            )

        self.function = function
        self.bins = Bins(bounds=bounds, bin_width=bin_width)
        self.counts = np.zeros_like(self.bins.centers)
        self.weight_function = weight_function

    def update(self) -> None:
        """
        Add values to the histogram.
        """
        values = self.function()
        if self.weight_function is None:
            weights = None
        else:
            weights = self.weight_function()
        counts, _ = np.histogram(values, bins=self.bins.edges, weights=weights)
        self.counts += counts

    def finalize(self, start: int, stop: int, step: int):
        pass

    def get(self, centers: bool = False) -> Tuple[np.ndarray, np.ndarray]:
        """
        Return the counts and the bin edges.

        :parameter centers: optional, return the bin centers instead of the bin edges
        :return: the counts and bin the edges/centers
        """
        if centers:
            return self.counts.copy(), self.bins.centers.copy()
        else:
            return self.counts.copy(), self.bins.edges.copy()


class Histogram2D(Analysis):
    """A two-dimensional, running histogram."""

    def __init__(
        self,
        x_function: StructureFunction,
        x_bounds: Tuple[float, float],
        x_bin_width: float,
        y_function: StructureFunction,
        y_bounds: Tuple[float, float],
        y_bin_width: float,
        weight_function: StructureFunction = None,
    ) -> None:
        """
        :param x_function: the function to be evaluated for the x-channel
        :param x_bounds: the lower and upper bounds of the x-channel
        :param x_bin_width: the bin width of the x-channel
        :param y_function: the function to be evaluated for the y-channel
        :param y_bounds:  the lower and upper bounds of the y-channel
        :param y_bin_width: the bin width of the y-channel
        """
        if x_function.universe != y_function.universe:
            raise MDStuffError(f"x- and y-functions work with different universes")
        if (
            weight_function is not None
            and weight_function.universe != x_function.universe
        ):
            raise MDStuffError(f"weight function works with a different universe")
        if len(x_function.shape) != 1:
            raise MDStuffError(f"x-function does not return a one-dimensional array")
        if len(y_function.shape) != 1:
            raise MDStuffError(f"y-function does not return a one-dimensional array")
        if weight_function is not None and len(weight_function.shape) != 1:
            raise MDStuffError(
                f"weight function does not return a one-dimensional array"
            )
        if x_function.shape != y_function.shape:
            raise MDStuffError(f"x- and y-function return different number of values")
        if weight_function is not None and weight_function.shape != x_function.shape:
            raise MDStuffError(f"weight function returns different number of values")

        super().__init__(universe=x_function.universe)

        self.x_function = x_function
        self.x_bins = Bins(bounds=x_bounds, bin_width=x_bin_width)
        self.y_function = y_function
        self.y_bins = Bins(bounds=y_bounds, bin_width=y_bin_width)
        self.counts = np.zeros((self.x_bins.nr, self.y_bins.nr))
        self.weight_function = weight_function

    def update(self,) -> None:
        """
        Add values to the histogram.
        """
        x_values = self.x_function()
        y_values = self.y_function()
        if self.weight_function is None:
            weights = None
        else:
            weights = self.weight_function()
        counts, *_ = np.histogram2d(
            x=x_values,
            y=y_values,
            bins=(self.x_bins.edges, self.y_bins.edges),
            weights=weights,
        )
        self.counts += counts

    def finalize(self, start: int, stop: int, step: int):
        pass

    def get(self, centers: bool = False) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Return the counts and the bin edges.

        :parameter centers: optional, return the bin centers instead of the bin edges
        :return: the counts and the bin edges/centers
        """
        if centers:
            return (
                self.counts.T.copy(),
                self.x_bins.centers.copy(),
                self.y_bins.centers.copy(),
            )
        else:
            return (
                self.counts.T.copy(),
                self.x_bins.edges.copy(),
                self.y_bins.edges.copy(),
            )
