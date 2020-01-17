from typing import Union, Tuple, Sequence

import MDAnalysis as mda

from .readers import ContinuousDCDReader


class NAMDUniverse(mda.Universe):
    """
    A NAMD specialized MDAnalysis.Universe.
    """

    def __init__(self, psf: str, dcds: Union[Sequence[str], Sequence[Tuple[str, int]]]):
        super(NAMDUniverse, self).__init__(psf, topology_format="PSF")

        filenames = [dcd[0] if isinstance(dcd, tuple) else dcd for dcd in dcds]
        lengths = [dcd[1] if isinstance(dcd, tuple) else None for dcd in dcds]
        reader = ContinuousDCDReader(filenames, lengths)
        if len(self.atoms) != reader.n_atoms:
            raise ValueError(
                "The number of atoms in the PSF and DCD files do not match"
            )
        self.trajectory = reader
