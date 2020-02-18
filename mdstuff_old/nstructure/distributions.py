from typing import Tuple

import numpy as np

from .functions import StructureFunction
from ..core.analyses import RunningHist
from ..core.errors import MDStuffError


class PDens(RunningHist):
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

        super().__init__(bounds=bounds, bin_width=bin_width)

        self.function = function
        self.mode = mode

    def update(self, values: np.ndarray = None, weights: np.ndarray = None) -> None:
        """
        Add values to the histogram.

        :param values: the values
        :param weights: optional, some weights
        """
        if values is None:
            values = self.function()
        super().update(values=values, weights=weights)

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
        if self.mode == "linear":
            counts /= self.bins.bin_width
        elif self.mode == "radial":
            rsqdr = (1.0 / 3.0) * (self.bins.edges[1:] ** 3 - self.bins.edges[:-1] ** 3)
            counts /= rsqdr
        if centers:
            return counts, self.bins.centers.copy()
        else:
            return counts, self.bins.edges.copy()
