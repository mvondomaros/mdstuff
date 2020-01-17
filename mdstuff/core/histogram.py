from typing import Tuple

import numpy as np

from .errors import MDStuffError


class Histogram:
    """
    A histogram class.
    """

    def __init__(
        self,
        lbound: float,
        ubound: float,
        nr_bins: int,
        normalize: bool = False,
        density: bool = False,
    ) -> None:
        """
        Set up a bounded histogram with a fixed number of bins.

        :param lbound: the lower histogram bound
        :param ubound: the upper histogram bound
        :param nr_bins: the number of bins
        :param normalize: optional, whether counts should be normalized
        :param density: optional, whether a density should be estimated
        """
        if nr_bins < 1:
            raise MDStuffError(f"invalid number of bins: {nr_bins}")

        self.bin_edges = np.linspace(lbound, ubound, nr_bins + 1)
        self.bin_centers = 0.5 * (self.bin_edges[1:] + self.bin_edges[:-1])
        self.bin_width = self.bin_edges[1] - self.bin_edges[0]
        self.total_counts = np.zeros_like(self.bin_centers)
        self.nr_additions = 0

        self.normalize = normalize
        self.density = density

    def add_array(self, a: np.ndarray, weights=None, norm: float = 1.0) -> None:
        """
        Add an array of values to the histogram.

        :param a: an array
        :param weights: optional, weights
        :param norm: optional, a normalization constant for the counts
        """
        counts, _ = np.histogram(
            a, bins=self.bin_edges, weights=weights, density=self.density
        )
        self.total_counts += counts * norm
        self.nr_additions += 1

    @property
    def counts(self) -> np.ndarray:
        """
        Return the counts.

        :return: the counts
        """
        if self.normalize:
            return self.total_counts / self.nr_additions
        else:
            return self.total_counts

    def to_tuple(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Return the bin centers and counts as a tuple.

        :return: a (bin centers, counts) tuple of numpy arrays
        """
        return self.bin_centers, self.counts
