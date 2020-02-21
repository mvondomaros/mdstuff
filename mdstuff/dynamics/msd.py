from typing import Tuple

import numpy as np
import tqdm
from MDAnalysis.core.groups import AtomGroup

from .helpers import msd_fft
from ..core import Analysis
from ..core.errors import ParameterValueError


class SingleParticleMSD(Analysis):

    MODE = "OTA"

    def __init__(self, ag: AtomGroup, n: int = 10, m: int = 4):
        super().__init__(universe=ag.universe)

        if not isinstance(n, int) or n <= 0 or n > len(self.universe.trajectory):
            raise ParameterValueError(
                name="n",
                value=n,
                allowed_values="an int, with 0 < n <= len(trajectory)",
            )
        if not isinstance(m, int) or m <= 0 or m > n:
            raise ParameterValueError(
                name="m", value=m, allowed_values="an int, with 0 < m <= n"
            )

        self.ag = ag
        self.n = n
        self.m = m
        self.msds = []
        self.time = None

    def update(self):
        # Work is completely done in finalize().
        pass

    def finalize(self, start: int = None, stop: int = None, step: int = None):
        # Get the center of mass positions.
        r = np.array(
            [
                self.ag.center_of_mass()
                for _ in tqdm.tqdm(
                    self.universe.trajectory[start:stop:step],
                    desc="SingleParticleMSD trajectory loop",
                )
            ]
        )

        traj_len = len(r)
        block_len = traj_len // self.n
        corr_len = block_len // 4

        msds = []
        for i_block in range(self.n):
            start = i_block * block_len
            end = start + block_len
            msd = [msd_fft(r[start:end, i], length=corr_len) for i in range(3)]
            self.msds.append(np.sum(msd, axis=0))

        self.time = np.arange(corr_len) * self.universe.trajectory.dt * step

    def get(self) -> Tuple[np.ndarray, np.ndarray]:
        return np.array(self.msds), self.time
