from typing import Tuple

from .base import StructureFunction
from .helpers import apply_mic
from .. import MDStuffError


class Distance(StructureFunction):
    def __init__(self, ag1, ag2, use_mic: bool = True):
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

    @property
    def shape(self) -> Tuple:
        return self._shape
