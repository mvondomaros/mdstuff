from typing import Sequence, Tuple

import MDAnalysis as mda
import numpy as np
import tqdm


def dprof(
    psf: str,
    dcds: Sequence[str],
    selection: str,
    lbound: float,
    ubound: float,
    nr_bins: int,
    start: int = 0,
    stop: int = -1,
    stride: int = 1,
    dimension: int = 2,
    center_selection: str = None,
    wrap: bool = False,
    mass_weighting: bool = False,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute a one-dimensional density profile.

    :param psf: the psf file
    :param dcds: a sequence of dcd files
    :param selection: the selection string
    :param lbound: the lower bound of the profile
    :param ubound: the upper bound of the profile
    :param nr_bins: the number of bins of the profile
    :param start: optional, the first time step of interest
    :param stop: optional, the last time step of interest
    :param stride: optional, the stride between time steps of interest
    :param dimension: optional, the dimension (x: 0, y: 1, z: 2) of the density profile
    :param center_selection: optional, the center selection string
    :param wrap: optional, whether wrapping should be applied
    :param mass_weighting: optional, whether mass densities should be computed instead of number densities
    :return: a (bin center, density) tuple of numpy arrays
    """
    # Set up the universe and select atoms.
    u = mda.Universe(psf, *dcds)
    s = u.select_atoms(selection)
    if center_selection:
        cs = u.select_atoms(center_selection)
    else:
        cs = None

    # Set up the histogram.
    bin_edges = np.linspace(lbound, ubound, nr_bins + 1)
    bin_width = bin_edges[1] - bin_edges[0]
    bin_centers = 0.5 * (bin_edges[1:] + bin_edges[:-1])
    counts = np.zeros_like(bin_centers)

    # Loop over all frames of interest.
    for nr_frames, ts in enumerate(tqdm.tqdm(u.trajectory[start:stop:stride])):
        # Center on center selection and wrap, if necessary.
        if cs:
            u.atoms.positions -= cs.atoms.center_of_mass()
        if wrap:
            # Add half a box first, then wrap, then subtract half a box again.
            # This way, the desired centering will be preserved.
            u.atoms.positions += 0.5 * ts.dimensions[:3]
            u.atoms.wrap()
            u.atoms.positions -= 0.5 * ts.dimensions[:3]

        # Compute the bin volume.
        volume = np.prod(ts.dimensions[:3]) / ts.dimensions[dimension] * bin_width

        # Update the histogram.
        x = s.atoms.positions[:, dimension]
        if mass_weighting:
            c, _ = np.histogram(x, bins=bin_edges, weights=s.atoms.masses)
        else:
            c, _ = np.histogram(x, bins=bin_edges)
        counts += c / volume

    # Normalize and return.
    counts /= nr_frames
    return bin_centers, counts
