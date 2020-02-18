
class DProf(Prof):
    """A one-dimensional density-profile."""

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

    def update(self, values: np.ndarray = None, weights: np.ndarray = None,) -> None:
        """
        Add weighted values to the histogram.

        :param values: the values for the x-channel
        :param weights: optional, some weights
        """
        volume = np.prod(self.universe.dimensions[:3]) / self.bins.nr
        if values is None:
            values = self.function()
        if weights is None:
            weights = self.weight_function() / volume
        super().update(values=values, weights=weights)
        self.nr_updates += 1
