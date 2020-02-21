from typing import List, Tuple

import numpy as np
import tqdm
from MDAnalysis.core.groups import AtomGroup

from .helpers import msd_fft
from ..core import Analysis, MDStuffError
from ..core.helpers import same_universe


class MSD(Analysis):

    MODE = "OTA"

    def __init__(self, ag_list: List[AtomGroup], length: int = None):
        if not same_universe(*ag_list):
            raise MDStuffError(
                f"the atom group list contains atom groups with different universes"
            )
        n = len(ag_list)
        if n == 0:
            raise MDStuffError(f"the atom group list is empty")
        super().__init__(universe=ag_list[0].universe)

        self.ag_list = ag_list
        self.length = len(self.universe.trajectory) // 2 if length is None else length
        self.msd = None
        self.time = None

    def update(self):
        # Work is completely done in finalize().
        pass

    def finalize(self, start: int, stop: int, step: int):
        n = len(self.ag_list)
        msds = []
        for i_loop, ag in enumerate(self.ag_list, start=1):
            # Get an array of positions
            r = []
            for ft_sq in tqdm.tqdm(
                self.universe.trajectory[start:stop:step],
                desc=f"MSD trajectory loop {i_loop}/{n}",
                leave=False,
            ):
                r.append(ag.center_of_mass())
            r = np.array(r)

            msds.append(
                np.sum([msd_fft(r[:, i], length=self.length) for i in range(3)], axis=0)
            )

        self.msd = np.mean(msds, axis=0)
        self.time = np.arange(self.msd.size) * self.universe.trajectory.dt * step

    def get(self) -> Tuple[np.ndarray, np.ndarray]:
        return self.msd, self.time
