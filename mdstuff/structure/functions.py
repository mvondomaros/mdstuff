from typing import Tuple

from MDAnalysis.core.groups import AtomGroup

from .base import StructureFunction
from .helpers import apply_mic
from .. import MDStuffError


# TODO: Comments.
class Distance(StructureFunction):
    def __init__(self, ag1: AtomGroup, ag2: AtomGroup, use_mic: bool = True):
        n1 = len(ag1)
        n2 = len(ag2)
        if n1 != n2:
            raise MDStuffError(
                f"the number of atoms in ag1 ({n1}) does not match the number of atoms in ag2 ({n2})"
            )
        elif n1 == 0:
            raise MDStuffError(f"ag1 and ag2 are empty")

        super().__init__(ag1.universe)

        self.ag1 = ag1
        self.ag2 = ag2
        self.use_mic = use_mic

        self._shape = (n1, 3)

    def __call__(self, *args, **kwargs):
        d = self.ag2.positions - self.ag1.positions
        if self.use_mic:
            apply_mic(d, self.universe.dimensions[:3])
        return d


class Position(StructureFunction):
    def __init__(self, ag: AtomGroup, use_pbc: bool = False):
        n = len(ag)
        if n == 0:
            raise MDStuffError(f"ag is empty")

        super().__init__(ag.universe)

        self.ag = ag
        self.use_pbc = use_pbc

        self._shape = (n, 3)

    def __call__(self, *args, **kwargs):
        x = self.ag.positions
        if self.use_pbc:
            apply_mic(x, self.universe.dimensions[:3])
        return x


class Mass(StructureFunction):
    def __init__(self, ag: AtomGroup):
        n = len(ag)
        if n == 0:
            raise MDStuffError(f"ag is empty")

        super().__init__(ag.universe)

        self.ag = ag

        self._shape = (n,)

    def __call__(self):
        return self.ag.masses


class Charge(StructureFunction):
    def __init__(self, ag: AtomGroup):
        n = len(ag)
        if n == 0:
            raise MDStuffError(f"ag is empty")

        super().__init__(ag.universe)

        self.ag = ag

        self._shape = (n,)

    def __call__(self):
        return self.ag.charges
