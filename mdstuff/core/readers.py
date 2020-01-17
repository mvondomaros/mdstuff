from typing import Sequence

import MDAnalysis as mda
import numpy as np


class ContinuousDCDReader(mda.coordinates.chain.ChainReader):
    """
    A continuous DCD reader.
    """

    format = "CONTINUOUSDCD"

    def __init__(self, filenames: Sequence[str], lengths: Sequence[int], **kwargs):
        # We're overwriting what happens in the ChainReader initialized, but we need to eplicitly call the
        # grand-parent initializer.
        super(mda.coordinates.chain.ChainReader, self).__init__()

        self.readers = [
            mda.coordinates.DCD.DCDReader(filename=f, **kwargs) for f in filenames
        ]
        self.filenames = np.array(filenames)
        self.__active_reader_index = 0
        self.n_atoms = self._get_same("n_atoms")
        n_total_frames = self._get("n_frames")
        n_frames = [l if l is not None else n for l, n in zip(lengths, n_total_frames)]
        self._start_frames = np.cumsum([0] + n_frames)
        self.n_frames = np.sum(n_frames)
        self.dts = np.array(self._get("dt"))
        self.total_times = self.dts * n_frames
        self.__chained_trajectories_iter = None
        self.ts = None
        self.rewind()

    @property
    def time(self):
        """Cumulative time of all frames in MDAnalysis time units (typically ps)."""
        trajindex, subframe = self._get_local_frame(self.frame)
        return self.total_times[:trajindex].sum() + (subframe + 1) * self.dts[trajindex]
