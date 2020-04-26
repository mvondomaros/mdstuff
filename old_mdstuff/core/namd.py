import os

import MDAnalysis
import collections
import numpy as np
import warnings
from typing import Sequence, Union, Tuple

from .base import Universe
from .errors import MDStuffError, ParameterValueError, DataError


# noinspection PyMissingConstructor
class ContinuousDCDReader(MDAnalysis.coordinates.chain.ChainReader):
    """
    A continuous DCD reader. The implementation is based on the MDAnalysis ChainReader.
    Differences are:
        - only supports DCD files
        - trajectory lengths must be given as a sequence (None means using all frames)
        - enforces a uniform time step
        - no skip parameter
    """

    format = "CONTINUOUSDCD"

    def __init__(self, filenames: Sequence[str], lengths: Sequence[int], **kwargs):
        """
        :param filenames: the sequence of file names
        :param lengths: the sequence of trajectory lengths
        """
        # We're overwriting what happens ChainReader.__init__(), so we need to explicitly call the
        # grand-parent initializer.
        super(MDAnalysis.coordinates.chain.ChainReader, self).__init__()

        # References to all readers and filenames.
        self.readers = []
        for f in filenames:
            try:
                self.readers.append(
                    MDAnalysis.coordinates.DCD.DCDReader(filename=f, **kwargs)
                )
            except OSError as e:
                warnings.warn(
                    f"ignoring corrupted DCD file '{f}' and any following files"
                )
                self.readers.pop()
                break
        self.filenames = np.array(filenames)

        # The index of the active reader.
        self.__active_reader_index = 0

        # We enforce an equal time step and equal number of atoms.
        self.dts = self._get("dt")  # Needed as an array by some inherited methods.
        self._get_same("dt")
        self.n_atoms = self._get_same("n_atoms")

        # Get the total number of frames.
        n_total_frames = self._get("n_frames")
        # Get the desired number of frames specified through lengths. Use all frames, if None is given.
        n_desired_frames = [
            length if length is not None else n
            for length, n in zip(lengths, n_total_frames)
        ]

        # The total number of desired frames.
        self.n_frames = np.sum(n_desired_frames)
        # The virtual indices corresponding to the start of each trajectory.
        self._start_frames = np.cumsum([0] + n_desired_frames)
        # The cumulative time passed after each individual trajectory.
        self.total_times = self.dt * np.array(n_desired_frames)
        # Technical stuff copied shamelessly from the base class.
        self.__chained_trajectories_iter = None
        self.ts = None
        self.rewind()

    @property
    def time(self) -> float:
        """
        Cumulative time of all frames in MDAnalysis time units (typically ps).

        :return: the time
        """
        traj_index, sub_frame = self._get_local_frame(self.frame)
        # Added +1 to sub_frame, since DCD files do not contain the initial coordinates.
        return (
            self.total_times[:traj_index].sum() + (sub_frame + 1) * self.dts[traj_index]
        )

    # Needed because of ABC requirements, but not implemented.
    def Writer(self, *args, **kwargs):
        raise NotImplementedError

    # Needed because of ABC requirements, but not implemented.
    @classmethod
    def parse_n_atoms(cls, *args, **kwargs):
        raise NotImplementedError

    @property
    def dt(self) -> float:
        """
        Returns the time step.

        :return: the time step
        """
        return self.dts[0]


class NAMDUniverse(Universe):
    """
    A NAMD specialized MDAnalysis Universe.

    Differences are:
        - works only with PSF and DCD files
        - multiple DCD files must be specified as a sequence
        - to account for overlap, the DCD sequence items may be tuples, where the second item is a maximum trajectory
          length
        - has an n_atoms field
    """

    def __init__(
        self, psf: str, dcd: Union[str, Sequence[str], Sequence[Tuple[str, int]]]
    ):
        """
        :param psf: the PSF file
        :param dcd: the DCD file name, or a Sequence of DCD file names, or a Sequence of (filename, length) tuples, or a
            mix thereof
        """
        # Check that the PSF file exists.
        if not os.path.exists(psf):
            raise DataError(f"psf file does not exist: {psf}")
        super().__init__(psf, topology_format="PSF")

        # Figure out the DCD files/length combo and initialize a ContinuousDCDReader.
        if isinstance(dcd, str):
            # One DCD file specified.
            filenames = [dcd]
            lengths = [None]
        elif isinstance(dcd, collections.Sequence):
            if isinstance(dcd, tuple) and len(dcd) == 2:
                if isinstance(dcd[1], int):
                    # One DCD file and one length specified.
                    filenames = [dcd[0]]
                    lengths = [dcd[1]]
                else:
                    # Two DCD files specified.
                    filenames = [dcd[0], dcd[1]]
                    lengths = [None, None]
            else:
                filenames = [dcd[0] if isinstance(dcd, tuple) else dcd for dcd in dcd]
                lengths = [dcd[1] if isinstance(dcd, tuple) else None for dcd in dcd]
        else:
            raise ParameterValueError(
                name="dcd",
                value=dcd,
                allowed_values="a str/tuple/sequence of tuples (see doc-string for details)",
            )
        if len(filenames) == 0:
            raise ParameterValueError(
                name="dcd",
                value=dcd,
                allowed_values="a str/tuple/sequence of tuples (see doc-string for details)",
            )
        # Check that the DCD files exists.
        for f in filenames:
            if not os.path.exists(f):
                raise DataError(f"dcd file does not exist: {f}")
        reader = ContinuousDCDReader(filenames, lengths)

        # Compare the number of atoms
        self.n_atoms = len(self.atoms)
        if self.n_atoms != reader.n_atoms:
            raise MDStuffError(
                f"The number of atoms in the PSF and DCD files do not match ({self.n_atoms} != {reader.n_atoms})"
            )
        self.trajectory = reader

    # Needed because of ABC requirements, but not implemented.
    def __setstate__(self, state):
        raise NotImplementedError

    # Needed because of ABC requirements, but not implemented.
    def __getstate__(self):
        raise NotImplementedError