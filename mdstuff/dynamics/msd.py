from collections import deque

import MDAnalysis
import numpy as np
import tqdm

from mdstuff.core.errors import MDStuffError
from mdstuff.tools.misc import aslist


def single_particle_msd(
    universe: MDAnalysis.Universe, selection: str, skip: int = 0, length: int = None,
) -> np.ndarray:
    """
    Compute the mean squared displacement (MSD) of a single particle.

    :param universe: an MDAnalysis Universe
    :param selection: the particle selection string
    :param skip: optional, skip this number of trajectory steps before starting the computation
    :param length: optional, the length of the MSD time series in trajectory steps; defaults to 2/3 of the trajectory
    :return: a (length, 2) numpy array containing time and MSD columns
    """
    if length is None:
        length = 2 * len(universe.trajectory) // 3
    else:
        if length > len(universe.trajectory):
            raise MDStuffError(
                f"length {length} exceeds the trajectory length of {len(universe.trajectory)}"
            )

    # Select atoms.
    sel = universe.select_atoms(selection)
    if sel.n_atoms == 0:
        raise MDStuffError(f"empty selection `{selection}`")

    # Allocate arrays for time and total squared displacement.
    time = np.arange(length) * universe.trajectory.dt
    tsd = np.zeros(length)

    # A buffer containing the last positions.
    x0 = deque(maxlen=length)

    # Loop over all frames.
    for frame in tqdm.tqdm(universe.trajectory):
        x = sel.atoms.center_of_mass()
        x0.appendleft(x)
        # Compute and accumulate the squared displacement.
        sd = np.sum((x - x0) ** 2, axis=-1)
        tsd[: sd.size] += sd

    # Normalize.
    msd = tsd / (
        np.full(length, fill_value=universe.trajectory.n_frames) - np.arange(length)
    )

    return np.column_stack([time, msd])
