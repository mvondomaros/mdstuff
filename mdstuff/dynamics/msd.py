import numpy as np

from mdstuff.core import SerialAnalysis, CompoundGroup, Universe
import tqdm

from .helpers import msd_fft


class MSD(SerialAnalysis):
    def __init__(
        self, cg: CompoundGroup, n: int = 100, start=None, stop=None, step=None
    ):
        self.cg = cg
        self.n = n
        self.start = start
        self.stop = stop
        self.step = step
        self.time = None
        self.msd = None
        self.n = 0

    def work(self, universe: Universe):
        start, stop, step = slice(self.start, self.stop, self.step).indices(
            len(universe.trajectory)
        )
        corr_len = ((stop - start) // step) // self.n
        self.time = np.arange(corr_len) * universe.trajectory.dt * step
        self.msd = np.zeros_like(self.time)

        for i, cg in enumerate(self.cg):
            # Get the center of mass positions.
            r = np.array(
                [
                    cg.center_of_mass()
                    for _ in tqdm.tqdm(
                        universe.trajectory[start:stop:step],
                        desc=f"MSD loop {i}/{len(self.cg)}",
                    )
                ]
            )

            for i in range(3):
                self.msd += msd_fft(r[:, i], length=corr_len)
                self.n += 1

    def save(self, name: str):
        np.savez(
            name, time=self.time, msd=self.msd / self.n,
        )
