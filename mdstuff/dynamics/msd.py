from typing import Tuple

import numpy as np
import tqdm
from MDAnalysis.core.groups import AtomGroup

from .helpers import msd_fft
from ..core import Analysis
from ..core.errors import ParameterValueError


class SingleParticleMSD(Analysis):

    MODE = "OTA"

    def __init__(self, ag: AtomGroup, n: int = 10, m: int = 2):
        super().__init__(universe=ag.universe)

        if not isinstance(n, int) or n <= 0:
            raise ParameterValueError(
                name="n", value=n, allowed_values="an int > 0",
            )
        if not isinstance(m, int) or m <= 0:
            raise ParameterValueError(name="m", value=m, allowed_values="an int > 0")

        self.ag = ag
        self.n = n
        self.m = m
        self.msds = []
        self.time = None

    def update(self):
        # Work is completely done in finalize().
        pass

    # TODO: rework start stop step
    def finalize(self, start: int = None, stop: int = None, step: int = None):
        # Get normalized start, stop, step values.
        start, stop, step = slice(start, stop, step).indices(
            len(self.universe.trajectory)
        )

        # Get the trajectory length, the block length, and the correlation length, taking start, stop, and step into
        # account.
        traj_len = (stop - start) // step
        block_len = traj_len // self.n
        corr_len = block_len // self.m

        if block_len == 0:
            raise ParameterValueError(
                name="start, stop, step, n",
                value=(start, stop, step, self.n),
                allowed_values="(stop-start)/step >= n (nr of blocks)",
            )

        for i_block in range(self.n):
            start = i_block * block_len * step
            stop = start + block_len * step
            # Get the center of mass positions.
            r = np.array(
                [
                    self.ag.center_of_mass()
                    for _ in tqdm.tqdm(
                        self.universe.trajectory[start:stop:step],
                        desc=f"MSD trajectory loop {i_block+1}/{self.n}",
                    )
                ]
            )
            msd = [msd_fft(r[:, i], length=corr_len) for i in range(3)]
            self.msds.append(np.sum(msd, axis=0))

        self.time = np.arange(corr_len) * self.universe.trajectory.dt * step

    def get(self) -> Tuple[np.ndarray, np.ndarray]:
        return np.array(self.msds), self.time
