from typing import Tuple

import numpy as np

from ..core import Analysis, Universe, MDStuffError
from MDAnalysis.core.groups import AtomGroup
import MDAnalysis.topology.tables


class FreeVolumeProfile(Analysis):
    def __init__(
        self, ag: AtomGroup, bin_width: float, dimension: int = 2, radii: dict = None,
    ):
        super().__init__(universe=ag.universe)

        self.ag = ag
        self.bin_width = bin_width
        self.dimension = dimension

        # Construct a dictionary of default radii and update it with user-provided values.
        self.radii = {
            "H": 1.2,
            "C": 1.7,
            "O": 1.52,
        }
        if radii is not None:
            self.radii.update(radii)

        # Check if the elements topology attribute is set.
        if not hasattr(self.universe, "elements"):
            self.universe.add_TopologyAttr("element")
            rounded_mass_to_element = {
                round(mass): element
                for element, mass in MDAnalysis.topology.tables.masses.items()
            }
            for atom in self.ag:
                atom.element = rounded_mass_to_element[np.round(atom.mass)]

        # Determine all elements that are present in the system.
        elements = np.unique([atom.element for atom in ag])

        # Construct a dictionary of 3D masks, corresponding to each element present in the system.
        try:
            self.masks = {
                element: self._create_mask(self.radii[element], self.bin_width)
                for element in elements
            }
        except KeyError as e:
            raise MDStuffError(f"unknown vdW radius: {e}")

        # Create the grid and the profile.
        self.box = np.copy(self.universe.dimensions[:3])
        self._create_grid()
        self.profile_centers = (
            0.5 * self.edges[self.dimension][1:] + 0.5 * self.edges[self.dimension][:-1]
        )
        self.profile = np.zeros_like(self.profile_centers)
        self.nr_updates = 0

    @staticmethod
    def _create_mask(
        radius: float, bin_width: float
    ) -> Tuple[Tuple[np.ndarray, np.ndarray, np.ndarray], np.ndarray]:
        xlo = np.floor(-radius / bin_width)
        xhi = np.ceil(radius / bin_width)
        xi = np.arange(xlo, xhi, dtype=np.intp)

        ylo = np.floor(-radius / bin_width)
        yhi = np.ceil(radius / bin_width)
        yi = np.arange(ylo, yhi, dtype=np.intp)

        zlo = np.floor(-radius / bin_width)
        zhi = np.ceil(radius / bin_width)
        zi = np.arange(zlo, zhi, dtype=np.intp)

        mask = (
            np.linalg.norm(
                [
                    (xi[:, None, None] + 0.5) * bin_width,
                    (yi[None, :, None] + 0.5) * bin_width,
                    (zi[None, None, :] + 0.5) * bin_width,
                ]
            )
            <= radius
        ).astype(np.int)

        return (xi, yi, zi), mask

    def _create_grid(self):
        nx = int(np.ceil(self.universe.dimensions[0] / self.bin_width))
        xe = np.linspace(0.0, self.universe.dimensions[0], nx + 1)
        dx = xe[1] - xe[0]

        ny = int(np.ceil(self.universe.dimensions[1] / self.bin_width))
        ye = np.linspace(0.0, self.universe.dimensions[1], ny + 1)
        dy = ye[1] - ye[0]

        nz = int(np.ceil(self.universe.dimensions[2] / self.bin_width))
        ze = np.linspace(0.0, self.universe.dimensions[2], nz + 1)
        dz = ze[1] - ze[0]

        self.edges = xe, ye, ze
        self.deltas = dx, dy, dz
        self.nr_bins = nx, ny, nz

    def update(self):
        if not np.all(np.isclose(self.box, self.universe.dimensions[:3])):
            raise MDStuffError(f"box size changed during free volume profile analysis")

        grid = np.zeros(self.nr_bins, dtype=np.int)

        for element, position in zip(
            self.ag.elements, self.ag.wrap()
        ):
            self._apply_mask(grid, element, position)

        # Compute the free volume ratio.
        grid = np.swapaxes(grid, self.dimension, 0)
        nz = np.sum(np.count_nonzero(grid, axis=self.dimension), axis=1)
        fvr = 1.0 - nz / grid.size * grid.shape[0]
        self.profile += fvr
        self.nr_updates += 1

    def _apply_mask(self, grid, element, position):
        shifts = self._get_shifts(position)
        mask_indices, mask = self.masks[element]
        xi = np.mod(mask_indices[0] + shifts[0], grid.shape[0])
        yi = np.mod(mask_indices[1] + shifts[1], grid.shape[1])
        zi = np.mod(mask_indices[2] + shifts[2], grid.shape[2])
        grid[np.ix_(xi, yi, zi)] += mask

    def _get_shifts(self, r):
        x, y, z = r
        xe, ye, ze = self.edges
        xi = np.searchsorted(xe, x, side="right") - 1
        yi = np.searchsorted(ye, y, side="right") - 1
        zi = np.searchsorted(ze, z, side="right") - 1
        return xi, yi, zi

    def finalize(self):
        pass

    def get(self) -> Tuple[np.ndarray, np.ndarray]:
        profile = self.profile.copy()
        if self.nr_updates > 0:
            profile /= self.nr_updates
        return profile, self.profile_centers
