from collections import deque
from typing import Sequence, Tuple

import MDAnalysis as mda
import numpy as np
import tqdm


def single_particle_msd(
    psf: str, dcds: Sequence[str], selection: str, maxlen: int = None, skip: int = 0
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute the mean squared displacement of a single particle.

    :param psf: the psf file
    :param dcds: a sequence of dcd files
    :param selection: the particle selection string
    :param maxlen: optional, the maximum number of trajectory steps for which the MSD will be evaluated
    :param skip: optional, skip this number of trajectory steps before starting the computation
    :return: a (time, MSD) tuple of numpy arrays
    """
    # FIXME: DCDs might overlap.
    # Set up the universe and select atoms.
    u = mda.Universe(psf, *dcds)
    s = u.select_atoms(selection)

    # Determine the number of trajectory time steps (maxlen) if it is not given (half the trajectory size).
    if maxlen is None:
        maxlen = (len(u.trajectory) - 2) // 2

    # Preallocate arrays.
    time = np.arange(maxlen) * u.trajectory[0].dt
    total_squared_displacement = np.zeros(maxlen)

    # Store the last positions in a deque.
    x0 = deque(maxlen=maxlen)

    # Loop over each time step and sum up the squared displacements to all relevant previous time steps.
    for ts in tqdm.tqdm(u.trajectory[skip:]):
        x = s.atoms.center_of_mass()
        x0.appendleft(x)
        squared_displacements = np.sum((x - x0) ** 2, axis=-1)
        total_squared_displacement[
            : squared_displacements.size
        ] += squared_displacements

    # Normalize.
    mean_squared_displacement = total_squared_displacement / (
        np.full(maxlen, fill_value=len(u.trajectory)) - np.arange(maxlen)
    )

    return time, mean_squared_displacement
