from __future__ import annotations

from typing import Tuple

import numpy as np

from .bins import Bins
from .universe import Analysis


class RunningHist(Analysis):
    """A one-dimensional running histogram."""

    def __init__(self, bounds: Tuple[float, float], bin_width: float) -> None:
        """
        :param bounds: the lower and upper bounds
        :param bin_width: the bin width
        """
        super().__init__()

        self.bins = Bins(bounds=bounds, bin_width=bin_width)
        self.counts = np.zeros_like(self.bins.centers)

    def update(self, values: np.ndarray, weights: np.ndarray = None) -> None:
        """
        Add values to the histogram.

        :param values: the values
        :param weights: optional, some weights
        """
        super().update()
        if weights is None:
            counts, _ = np.histogram(values, bins=self.bins.edges)
        else:
            counts, _ = np.histogram(values, bins=self.bins.edges, weights=weights)
        self.counts += counts

    def get(self, centers: bool = False,) -> Tuple[np.ndarray, np.ndarray]:
        """
        Return the counts and the bins edges.

        :parameter centers: optional, return the bin centers instead of the bin edges
        :return: the counts and bin the edges/centers
        """
        if centers:
            return self.counts.copy(), self.bins.centers.copy()
        else:
            return self.counts.copy(), self.bins.edges.copy()


class RunningHist2D(Analysis):
    """A two-dimensional running histogram."""

    def __init__(
        self,
        x_bounds: Tuple[float, float],
        x_bin_width: float,
        y_bounds: Tuple[float, float],
        y_bin_width: float,
    ) -> None:
        """
        :param x_bounds: the lower and upper bounds of the x-channel
        :param x_bin_width: the bin width of the x-channel
        :param y_bounds:  the lower and upper bounds of the y-channel
        :param y_bin_width: the bin width of the y-channel
        """
        super().__init__()

        self.x_bins = Bins(bounds=x_bounds, bin_width=x_bin_width)
        self.y_bins = Bins(bounds=y_bounds, bin_width=y_bin_width)
        self.counts = np.zeros((self.x_bins.nr, self.y_bins.nr))

    def update(
        self, x_values: np.ndarray, y_values: np.ndarray, weights: np.ndarray = None
    ) -> None:
        """
        Add values to the histogram.

        :param x_values: the values of the x-channel
        :param y_values: the values of the y-channel
        :param weights: optional, some weights
        """
        if weights is None:
            counts, *_ = np.histogram2d(
                x=x_values, y=y_values, bins=(self.x_bins.edges, self.y_bins.edges)
            )
        else:
            counts, *_ = np.histogram2d(
                x=x_values,
                y=y_values,
                bins=(self.x_bins.edges, self.y_bins.edges),
                weights=weights,
            )
        self.counts += counts

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
