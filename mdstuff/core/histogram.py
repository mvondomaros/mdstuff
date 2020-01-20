import numpy as np

from .errors import MDStuffError


class Histogram:
    """
    A histogram class.
    """

    def __init__(
        self, lbound: float, ubound: float, nr_bins: int, accumulate: bool = False
    ) -> None:
        """
        Set up a bounded histogram with a fixed number of bins.

        :param lbound: the lower histogram bound
        :param ubound: the upper histogram bound
        :param nr_bins: the number of bins
        :param accumulate: optional, whether multiple additions should be accumulated instead of averaged
        """
        if nr_bins < 1:
            raise MDStuffError(f"invalid number of bins: {nr_bins}")

        self.bin_edges = np.linspace(lbound, ubound, nr_bins + 1)
        self.bin_centers = 0.5 * (self.bin_edges[1:] + self.bin_edges[:-1])
        self.bin_width = self.bin_edges[1] - self.bin_edges[0]
        self.total_counts = np.zeros_like(self.bin_centers)
        self.nr_additions = 0
        self.accumulate = accumulate

    def add_array(self, a: np.ndarray, weights=None, factor: float = 1.0) -> None:
        """
        Add an array of values to the histogram.

        :param a: an array
        :param weights: optional, weights for each element in a
        :param factor: optional, a global normalization factor
        """
        counts, _ = np.histogram(a, bins=self.bin_edges, weights=weights)
        self.total_counts += counts * factor
        self.nr_additions += 1

    @property
    def counts(self) -> np.ndarray:
        """
        Return the counts.

        :return: a numpy array containing the counts
        """
        if self.accumulate:
            return self.total_counts
        else:
            return self.total_counts / self.nr_additions

    def to_array(self) -> np.ndarray:
        """
        Return as a numpy array.

        :return: a numpy array containing bin centers and counts
        """
        return np.column_stack([self.bin_centers, self.counts])
