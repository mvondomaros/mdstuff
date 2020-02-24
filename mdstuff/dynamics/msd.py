from typing import Tuple

import numpy as np
import tqdm
from MDAnalysis.core.groups import AtomGroup

from .helpers import msd_fft
from ..core import OneTimeAnalysis
from ..core.errors import ParameterValueError


class SingleParticleMSD(OneTimeAnalysis):
    def __init__(self, ag: AtomGroup, n: int = 100, start=None, stop=None, step=None):
        super().__init__(universe=ag.universe)

        self.start, self.stop, self.step = slice(start, stop, step).indices(
            len(self.universe.trajectory)
        )

        self.corr_len = ((self.stop - self.start) // self.step) // n
        if self.corr_len <= 0:
            raise ParameterValueError(
                name="n",
                value=n,
                allowed_values=f"an int, with 0 < n <= {(self.stop-self.start)//self.step}",
            )

        self.ag = ag
        self.msd = None
        self.time = None

    def update(self):
        # Work is completely done in finalize().
        pass

    def finalize(self):
        # Get the center of mass positions.
        r = np.array(
            [
                self.ag.center_of_mass()
                for _ in tqdm.tqdm(
                    self.universe.trajectory[self.start : self.stop : self.step],
                    desc=f"MSD trajectory loop",
                )
            ]
        )
        msd = [msd_fft(r[:, i], length=self.corr_len) for i in range(3)]

        self.msd = np.mean(msd, axis=0)
        self.time = np.arange(self.corr_len) * self.universe.trajectory.dt * self.step

    def get(self) -> Tuple[np.ndarray, np.ndarray]:
        return self.msd, self.time
