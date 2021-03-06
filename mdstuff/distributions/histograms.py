import numbers
from typing import Callable, Tuple

import numpy as np

from mdstuff.core import ParallelAnalysis, ParameterValueError


class Bins:
    """
    A one-dimensional bins class.
    """

    def __init__(self, bin_spec: Tuple[float]):
        """
        Initialize the bins.

        Args:
            bin_spec (Tuple[float]): The lower bound, the upper bound, the bin width.

        Raises:
            ParameterValueError: If a parameter takes on an invalid value.
        """
        if len(bin_spec) != 3:
            raise ParameterValueError(
                name="bin_spec", value=bin_spec, message="should be a 3-tuple"
            )
        for b in bin_spec:
            if not isinstance(b, numbers.Number):
                raise ParameterValueError(
                    name="bin_spec",
                    value=bin_spec,
                    message="should contain the lower bound, the upper bound, and the bin width",
                )

        if (
            isinstance(bin_spec[0], int)
            and isinstance(bin_spec[1], int)
            and isinstance(bin_spec[2], int)
            and ((bin_spec[1] - bin_spec[0]) % bin_spec[2]) == 0
        ):
            self.edges = np.array(
                list(range(bin_spec[0], bin_spec[1] + bin_spec[2], bin_spec[2]))
            )
        else:
            self.edges = np.arange(
                bin_spec[0], bin_spec[1] + 0.5 * bin_spec[2], bin_spec[2]
            )
            if not np.isclose(self.edges[-1], bin_spec[1]):
                raise ParameterValueError(
                    name="bin_spec", value=bin_spec, message="bounds/bin width mismatch"
                )

        if len(self.edges) < 2:
            raise ParameterValueError(
                name="bin_spec", value=bin_spec, message="no bins defined"
            )

        self.centers = 0.5 * (self.edges[1:] + self.edges[:-1])
        self.count = self.centers.size
        self.width = self.edges[1] - self.edges[0]


class Histogram(ParallelAnalysis):
    """
    A one-dimensional, continuously updating histogram.
    """

    def __init__(
        self,
        values: Callable,
        bins: Tuple,
        weights: Callable = None,
        filter: Callable = None,
    ):
        """
        Initialize a histogram.

        Args:
            values (Callable): The function to be called for getting values.
            bins (Tuple): The bin specificiation.
            weights (Callable, optional): Function to be called for weights. Defaults to None.
            filter (Callable, optional): The function to be called for getting a mask. Defaults to None.

        Raises:
            ParameterValueError: If a parameter takes on an invalid value.
        """
        super().__init__()

        if not callable(values):
            raise ParameterValueError(
                "values", value=values, message="should be callable"
            )
        self.values = values

        if weights is not None and not callable(weights):
            raise ParameterValueError(
                "weights", value=weights, message="should be callable"
            )
        self.weights = weights

        if filter is not None and not callable(filter):
            raise ParameterValueError(
                "filter", value=filter, message="should be callable"
            )
        self.filter = filter

        self.bins = Bins(bin_spec=bins)
        self.counts = np.zeros_like(self.bins.centers)
        self.nr_updates = 0

    def update(self):
        """
        Add values to the histogram.
        """
        values = self.values()
        if self.weights is None:
            weights = None
        else:
            weights = self.weights()

        if self.filter is not None:
            mask = self.filter()
            values = values[mask]
            if self.weights is not None:
                weights = weights[mask]

        counts, _ = np.histogram(values, bins=self.bins.edges, weights=weights)
        self.counts += counts
        self.nr_updates += 1

    def save(self, name: str, normalization: str = None):
        """
        Save the histogram.

        Args:
            name (str): The file name.
            normalization (str, optional): The normalization mode. Defaults to None.
                Supported modes are:
                    - "pdens": Compute a probability density.
                    - "ave": Compute averages.

        Raises:
            NotImplementedError: If an unsupported normalization mode is chosen.
        """
        if normalization is None:
            norm_counts = self.counts
        elif normalization == "pdens":
            norm_counts = self.counts / (np.sum(self.counts) * self.bins.width)
        elif normalization == "ave":
            norm_counts = self.counts / self.nr_updates
        else:
            raise NotImplementedError(f"normalization = {normalization}")

        np.savez(
            name,
            bin_centers=self.bins.centers,
            bin_edges=self.bins.edges,
            values=norm_counts,
            nr_updates=self.nr_updates,
        )


class Histogram2D(ParallelAnalysis):
    """
    A two-dimensional, continuously updating histogram.
    """

    def __init__(
        self,
        x_values: Callable,
        y_values: Callable,
        x_bins: Tuple,
        y_bins: Tuple,
        filter: Callable = None,
    ):
        """
        Initialize a histogram.

        Args:
            x_values (Callable): The function to be called for getting the x-values.
            y_values (Callable): The function to be called for getting the y-values.
            x_bins (Tuple): The x-bin specificiation.
            y_bins (Tuple): The y-bin specificiation.
            filter (Callable, optional): The function to be called for getting a mask. Defaults to None.

        Raises:
            ParameterValueError: If a parameter takes on an invalid value.
        """
        super().__init__()

        if not callable(x_values):
            raise ParameterValueError(
                "x_values", value=x_values, message="should be callable"
            )
        self.x_values = x_values

        if not callable(y_values):
            raise ParameterValueError(
                "y_values", value=y_values, message="should be callable"
            )
        self.y_values = y_values

        if filter is not None and not callable(filter):
            raise ParameterValueError(
                "filter", value=filter, message="should be callable"
            )
        self.filter = filter

        self.x_bins = Bins(bin_spec=x_bins)
        self.y_bins = Bins(bin_spec=y_bins)
        self.counts = np.zeros((self.x_bins.count, self.y_bins.count))
        self.nr_updates = 0

    def update(self):
        """
        Add values to the histogram.
        """
        x_values = self.x_values()
        y_values = self.y_values()
        if self.filter is not None:
            mask = self.filter()
            x_values = x_values[mask]
            y_values = y_values[mask]

        counts, *_ = np.histogram2d(
            x=x_values, y=y_values, bins=(self.x_bins.edges, self.y_bins.edges),
        )
        self.counts += counts
        self.nr_updates += 1

    def save(self, name: str, normalization: str = None):
        """
        Save the histogram.

        Args:
            name (str): The file name.
            normalization (str, optional): The normalization mode. Defaults to None.
                Supported modes are:
                    - "pdens": Compute a probability density.
                    - "ave": Compute averages.
                    - "cond_pdens": Compute a conditional probability density (with respect to the x-axis).

        Raises:
            NotImplementedError: If an unsupported normalization mode is chosen.
        """
        if normalization is None:
            norm_counts = self.counts
        elif normalization == "pdens":
            norm_counts = self.counts / (
                np.sum(self.counts) * self.x_bins.width * self.y_bins.width
            )
        elif normalization == "ave":
            norm_counts = self.counts / self.nr_updates
        elif normalization == "cond_pdens":
            norm_counts = np.copy(self.counts)
            norm = np.sum(norm_counts, axis=1)[:, None] * self.y_bins.width
            np.true_divide(norm_counts, norm, out=norm_counts, where=norm > 0.0)
        else:
            raise NotImplementedError(f"normalization = {normalization}")

        np.savez(
            name,
            x_bin_centers=self.x_bins.centers,
            x_bin_edges=self.x_bins.edges,
            y_bin_centers=self.y_bins.centers,
            y_bin_edges=self.y_bins.edges,
            values=norm_counts,
            nr_updates=self.nr_updates,
        )
