from typing import Sequence, Tuple, Union, List

import MDAnalysis
import numpy as np
import tqdm

from ..core import Histogram, MDStuffError


def dprof(
    universe: MDAnalysis.Universe,
    selections: Sequence[str],
    lbound: float,
    ubound: float,
    nr_bins: int,
    center_group: str = "all",
    direction: int = 2,
    mass_profile: bool = False,
    skip: int = 0,
    wrap: bool = False,
) -> Union[
    List[np.ndarray],
]:
    """
    Compute one or more density profiles.

    :param universe: an MDAnalysis Universe
    :param selections: a sequence of selection strings
    :param lbound: the lower bound of the profiles
    :param ubound: the upper bound of the profiles
    :param nr_bins: the number of bins in the profiles
    :param center_group: optional, the center selection string
    :param direction: optional, the direction (x: 0, y: 1, z: 2) of the profiles
    :param mass_profile: optional, whether mass density profiles should be computed
    :param skip: optional, skip this number of trajectory steps before starting the computation
    :param wrap: optional, whether wrapping should be performed
    :return: a list of of numpy arrays containing bin centers and densities
    """
    # Select atoms.
    atom_groups = []
    for ag in selections:
        ag = universe.select_atoms(ag)
        if ag.n_atoms == 0:
            raise MDStuffError(f"empty selection `{ag}`")
        atom_groups.append(ag)
    center_group = universe.select_atoms(center_group)

    # Set up the histograms.
    histograms = [Histogram(lbound, ubound, nr_bins) for _ in atom_groups]

    # Loop over all frames of interest.
    for frame in tqdm.tqdm(universe.trajectory[skip:]):
        # Move the COM of the center selection to the coordinate system origin.
        universe.atoms.positions -= center_group.center_of_mass()
        if wrap:
            # Add half a box first, then wrap, then subtract half a box again.
            # This way, the desired centering will be preserved.
            universe.atoms.positions += 0.5 * frame.dimensions[:3]
            universe.atoms.wrap()
            universe.atoms.positions -= 0.5 * frame.dimensions[:3]

        # Compute the current inverse volume of the bins.
        inverse_volume = frame.dimensions[direction] / (
            np.prod(frame.dimensions[:3]) * histograms[0].bin_width
        )

        # Update the histograms.
        for ag, h in zip(atom_groups, histograms):
            x = ag.positions[:, direction]
            if mass_profile:
                h.add_array(x, weights=ag.masses, factor=inverse_volume)
            else:
                h.add_array(x, factor=inverse_volume)

    # Return the histogram(s).
    return [h.to_array() for h in histograms]
