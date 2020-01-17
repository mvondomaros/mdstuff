from typing import Union, Tuple, Sequence

import MDAnalysis

from .errors import MDStuffError
from .readers import ContinuousDCDReader


class NAMDUniverse(MDAnalysis.Universe):
    """
    A NAMD specialized MDAnalysis Universe.
    Differences are:
        - works only with PSF and DCD files
        - DCD files must be specified as a sequence
        - to account for overlap, DCD sequence items may be tuples, where the second item is a maximum trajectory length
        - has an n_atoms field

    """

    def __init__(self, psf: str, dcds: Union[Sequence[str], Sequence[Tuple[str, int]]]):
        super(NAMDUniverse, self).__init__(psf, topology_format="PSF")
        # Save the number of atoms as a field.
        self.n_atoms = len(self.atoms)
        # Initialize a ContinuousDCDReader.
        filenames = [dcd[0] if isinstance(dcd, tuple) else dcd for dcd in dcds]
        lengths = [dcd[1] if isinstance(dcd, tuple) else None for dcd in dcds]
        reader = ContinuousDCDReader(filenames, lengths)
        # Compare the number of atoms
        if self.n_atoms != reader.n_atoms:
            raise MDStuffError(
                f"The number of atoms in the PSF and DCD files do not match ({self.n_atoms} != {reader.n_atoms})"
            )
        self.trajectory = reader

    def __setstate__(self, state):
        raise NotImplementedError

    def __getstate__(self):
        raise NotImplementedError
