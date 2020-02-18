# from __future__ import annotations
#
# from typing import Tuple
#
# import numpy as np
#
# from .functions import StructureFunction
# from ..core import MDStuffError
# from ..core.analyses import RunningHist, RunningHist2D
#
#
# class PDens(RunningHist):
#     """A one-dimensional probability density."""
#
#     def __init__(
#         self,
#         function: StructureFunction,
#         bounds: Tuple[float, float],
#         bin_width: float,
#         mode: str = "linear",
#     ) -> None:
#         """
#         :param function: a structure function
#         :param bounds: the lower and upper bounds of the histogram
#         :param bin_width: the bin width of the histogram
#         :param mode: optional, "linear" or "radial", the normalization scheme
#         """
#         super().__init__(bounds=bounds, bin_width=bin_width)
#
#         self.function = function
#         if mode not in ["linear", "radial"]:
#             raise MDStuffError(f"invalid mode: {mode}")
#         self.mode = mode
#
#     def _late_init(self) -> None:
#         super()._late_init()
#         self.function.set_universe(self.universe)
#
#     def update(self, values: np.ndarray = None, weights: np.ndarray = None) -> None:
#         """
#         Add values to the histogram.
#
#         :param values: the values
#         :param weights: optional, some weights
#         """
#         if values is None:
#             values = self.function()
#         super().update(values=values, weights=weights)
#
#     def get(self, centers: bool = False) -> Tuple[np.ndarray, np.ndarray]:
#         """
#         Return the probability densities and the bin edges.
#
#         :parameter centers: optional, return the bin centers instead of the bin edges
#         :return: the probability densities and the bin edges/centers
#         """
#         """
#         Return the probability densities and the bin edges.
#
#         :parameter centers: optional, return the bin centers instead of the bin edges
#         :return: the probability densities and the bin edges/centers
#         """
#
#         counts = self.counts.copy()
#         n = np.sum(counts)
#         if n > 0.0:
#             counts /= n
#         if self.mode == "linear":
#             counts /= self.bins.bin_width
#         elif self.mode == "radial":
#             rsqdr = (1.0 / 3.0) * (self.bins.edges[1:] ** 3 - self.bins.edges[:-1] ** 3)
#             counts /= rsqdr
#         if centers:
#             return counts, self.bins.centers.copy()
#         else:
#             return counts, self.bins.edges.copy()
#
#
# class PDens2D(RunningHist2D):
#     """A two-dimensional probability density."""
#
#     def __init__(
#         self,
#         x_function: StructureFunction,
#         x_bounds: Tuple[float, float],
#         x_bin_width: float,
#         y_function: StructureFunction,
#         y_bounds: Tuple[float, float],
#         y_bin_width: float,
#         x_mode: str = "linear",
#         y_mode: str = "linear",
#     ) -> None:
#         """
#         :param x_function: the structure function for the x-channel
#         :param x_bounds: the lower and upper bounds of the x-channel
#         :param x_bin_width: the bin width of the x-channel
#         :param y_function: the structure function for the y-channel
#         :param y_bounds:  the lower and upper bounds of the y-channel
#         :param y_bin_width: the bin width of the y-channel
#         :param x_mode: optional, "linear" or "radial", the normalization scheme for the x-channel
#         :param y_mode: optional, "linear" or "radial", the normalization scheme for the y-channel
#         """
#         super().__init__(
#             x_bounds=x_bounds,
#             x_bin_width=x_bin_width,
#             y_bounds=y_bounds,
#             y_bin_width=y_bin_width,
#         )
#         self.x_function = x_function
#         self.y_function = y_function
#         if x_mode not in ["linear", "radial"]:
#             raise MDStuffError(f"invalid x_mode: {x_mode}")
#         if y_mode not in ["linear", "radial"]:
#             raise MDStuffError(f"invalid y_mode: {y_mode}")
#         self.x_mode = x_mode
#         self.y_mode = y_mode
#
#     def _late_init(self) -> None:
#         super()._late_init()
#         self.x_function.set_universe(self.universe)
#         self.y_function.set_universe(self.universe)
#         if self.x_function.shape != self.y_function.shape:
#             raise MDStuffError(
#                 f"x- and y-functions must return the same number of values; "
#                 f"got {self.x_function.shape} and {self.y_function.shape}"
#             )
#
#     def update(
#         self,
#         x_values: np.ndarray = None,
#         y_values: np.ndarray = None,
#         weights: np.ndarray = None,
#     ) -> None:
#         """
#         Add values to the histogram.
#
#         :param x_values: the values for the x-channel
#         :param y_values: the values for the y-channel
#         :param weights: optional, some weights
#         """
#         if x_values is None:
#             x_values = self.x_function()
#         if y_values is None:
#             y_values = self.y_function()
#         super().update(x_values=x_values, y_values=y_values, weights=weights)
#
#     def get(self, centers: bool = False) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
#         """
#         Return the probability densities and the bin edges.
#
#         :parameter centers: optional, return the bin centers instead of the bin edges
#         :return: the probability densities and the bin edges/centers
#         """
#         counts = self.counts.T.copy()
#         n = np.sum(counts)
#         if n > 0.0:
#             counts /= n
#         if self.x_mode == "linear":
#             counts /= self.x_bins.bin_width
#         elif self.x_mode == "radial":
#             rsqdr = (1.0 / 3.0) * (
#                 self.x_bins.edges[1:] ** 3 - self.x_bins.edges[:-1] ** 3
#             )
#             counts /= rsqdr
#         else:
#             raise NotImplementedError
#         if self.y_mode == "linear":
#             counts /= self.y_bins.bin_width
#         elif self.y_mode == "radial":
#             rsqdr = (1.0 / 3.0) * (
#                 self.y_bins.edges[1:] ** 3 - self.y_bins.edges[:-1] ** 3
#             )
#             counts /= rsqdr
#         else:
#             raise NotImplementedError
#         if centers:
#             return counts, self.x_bins.centers.copy(), self.y_bins.centers.copy()
#         else:
#             return counts, self.x_bins.edges.copy(), self.y_bins.edges.copy()
#
#
# class CorrFunc2D(PDens2D):
#     """A two-dimensional correlation function."""
#
#     def get(self, centers: bool = False) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
#         """
#         Return the probability densities and the bin edges.
#
#         :parameter centers: optional, return the bin centers instead of the bin edges
#         :return: the probability densities and the bin edges/centers
#         """
#         # TODO: Might need fixing.
#         counts, _, _ = super().get()
#         hx = np.sum(counts, axis=0)
#         hy = np.sum(counts, axis=1)
#         hxy = np.tensordot(hy, hx, axes=0)
#         if self.x_mode == "linear":
#             counts /= self.x_bins.bin_width
#         elif self.x_mode == "radial":
#             rsqdr = (1.0 / 3.0) * (
#                 self.x_bins.edges[1:] ** 3 - self.x_bins.edges[:-1] ** 3
#             )
#             hxy /= rsqdr
#         else:
#             raise NotImplementedError
#         if self.y_mode == "linear":
#             counts /= self.y_bins.bin_width
#         elif self.y_mode == "radial":
#             rsqdr = (1.0 / 3.0) * (
#                 self.y_bins.edges[1:] ** 3 - self.y_bins.edges[:-1] ** 3
#             )
#             hxy /= rsqdr
#         else:
#             raise NotImplementedError
#         counts -= hxy
#         if centers:
#             return counts, self.x_bins.centers.copy(), self.y_bins.centers.copy()
#         else:
#             return counts, self.x_bins.edges.copy(), self.y_bins.edges.copy()
#
#
# class Prof(RunningHist):
#     """A one-dimensional profile."""
#
#     def __init__(
#         self,
#         function: StructureFunction,
#         bounds: Tuple[float, float],
#         bin_width: float,
#         weight_function: StructureFunction,
#     ) -> None:
#         """
#         :param function: the structure function that defines the histogram
#         :param bounds: the lower and upper bounds of the histogram
#         :param bin_width: the bin width of the histogram
#         :param weight_function: the structure function that computes the weights
#         """
#         super().__init__(
#             bounds=bounds, bin_width=bin_width,
#         )
#         self.function = function
#         self.weight_function = weight_function
#         self.nr_updates = 0
#
#     def _late_init(self) -> None:
#         super()._late_init()
#         self.function.set_universe(self.universe)
#         self.weight_function.set_universe(self.universe)
#
#     def update(self, values: np.ndarray = None, weights: np.ndarray = None,) -> None:
#         """
#         Add weighted values to the histogram.
#
#         :param values: the values for the x-channel
#         :param weights: optional, some weights
#         """
#         if values is None:
#             values = self.function()
#         if weights is None:
#             weights = self.weight_function()
#         super().update(values=values, weights=weights)
#         self.nr_updates += 1
#
#     def get(self, centers: bool = False) -> Tuple[np.ndarray, np.ndarray]:
#         """
#         Return the probability densities and the bins edges.
#
#         :parameter centers: optional, return the bin centers instead of the bin edges
#         :return: the probability densities and the bin edges/centers
#         """
#         counts = self.counts.copy()
#         if self.nr_updates > 0:
#             counts /= self.nr_updates
#         if centers:
#             return counts, self.bins.centers.copy()
#         else:
#             return counts, self.bins.edges.copy()
#
#
# class DProf(Prof):
#     """A one-dimensional density-profile."""
#
#     def __init__(
#         self,
#         function: StructureFunction,
#         bounds: Tuple[float, float],
#         bin_width: float,
#         weight_function: StructureFunction,
#     ) -> None:
#         """
#         :param function: the structure function that defines the histogram
#         :param bounds: the lower and upper bounds of the histogram
#         :param bin_width: the bin width of the histogram
#         :param weight_function: the structure function that computes the weights
#         """
#         super().__init__(
#             function=function,
#             bounds=bounds,
#             bin_width=bin_width,
#             weight_function=weight_function,
#         )
#
#     def update(self, values: np.ndarray = None, weights: np.ndarray = None,) -> None:
#         """
#         Add weighted values to the histogram.
#
#         :param values: the values for the x-channel
#         :param weights: optional, some weights
#         """
#         volume = np.prod(self.universe.dimensions[:3]) / self.bins.nr
#         if values is None:
#             values = self.function()
#         if weights is None:
#             weights = self.weight_function() / volume
#         super().update(values=values, weights=weights)
#         self.nr_updates += 1
