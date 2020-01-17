from typing import Sequence

import MDAnalysis
import numpy as np


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
        # We're overwriting what happens in the ChainReader initialized, but we need to eplicitly call the
        # grand-parent initializer.
        super(MDAnalysis.coordinates.chain.ChainReader, self).__init__()

        # References to all readers and filenames.
        self.readers = [
            MDAnalysis.coordinates.DCD.DCDReader(filename=f, **kwargs)
            for f in filenames
        ]
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
            l if l is not None else n for l, n in zip(lengths, n_total_frames)
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
    def time(self):
        """Cumulative time of all frames in MDAnalysis time units (typically ps)."""
        traj_index, sub_frame = self._get_local_frame(self.frame)
        # Added +1 to sub_frame, since DCD files do not contain the initial coordinates.
        return (
            self.total_times[:traj_index].sum() + (sub_frame + 1) * self.dts[traj_index]
        )

    def Writer(self, filename, **kwargs):
        raise NotImplementedError

    @classmethod
    def parse_n_atoms(cls, filename, **kwargs):
        raise NotImplementedError

    @property
    def dt(self):
        return self.dts[0]
