import MDAnalysis
import numpy as np
from MDAnalysis.core.groups import AtomGroup
from typing import List

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


class Axis3(StructureFunction):
    def __init__(self, ag1: AtomGroup, ag2: AtomGroup, ag3: AtomGroup, ag4: AtomGroup):
        n1 = len(ag1)
        n2 = len(ag2)
        n3 = len(ag3)
        n4 = len(ag4)
        if n1 != n2 != n3 != n4:
            raise MDStuffError(
                f"the number of atoms in ag1 ({n1}), ag2 ({n2}), ag3 ({n3}), and ag4 ({n4}) do not match"
            )
        elif n1 == 0:
            raise MDStuffError(f"all atom groups are empty")

        super().__init__(ag1.universe)

        self.ag1 = ag1
        self.ag2 = ag2
        self.ag3 = ag3
        self.ag4 = ag4

        self._shape = (n1, 3)

    def __call__(self, *args, **kwargs):
        axis = (
            self.ag2.positions
            + self.ag3.positions
            + self.ag4.positions
            - 3 * self.ag1.positions
        )
        return axis


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


class CompoundMass(StructureFunction):
    def __init__(self, ag_list: List[AtomGroup]):
        n = len(ag_list)
        if n == 0:
            raise MDStuffError(f"no compounds selected")

        super().__init__(ag_list[0].universe)

        self.ag_list = ag_list

        self._shape = (n,)

    def __call__(self):
        return [np.sum(ag.masses) for ag in self.ag_list]


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


class Angle(StructureFunction):
    """
    Compute the angle between ag1-ag2-ag3
    """

    def __init__(
        self, ag1: AtomGroup, ag2: AtomGroup, ag3: AtomGroup, use_mic: bool = False,
    ):
        n1 = len(ag1)
        n2 = len(ag2)
        n3 = len(ag3)
        if not n1 == n2 == n3:
            raise MDStuffError(
                f"the number of atoms in groups 1 ({n1}), 2 ({n2}), and 3 ({n3}) do not match"
            )
        elif n1 == 0:
            raise MDStuffError(f"all atom groups are empty")

        super().__init__(ag2.universe)

        self.ag1 = ag1
        self.ag2 = ag2
        self.ag3 = ag3
        if use_mic:
            self.box = self.universe.dimensions
        else:
            self.box = None

        self._shape = (n1,)

    def __call__(self, *args, **kwargs):
        a = MDAnalysis.lib.distances.calc_angles(
            self.ag1.positions,
            self.ag2.positions,
            self.ag3.positions,
            box=self.box,
            backend="OpenMP",
        )
        a *= 180.0 / np.pi
        return a


class Dihedral(StructureFunction):
    def __init__(
        self,
        ag1: AtomGroup,
        ag2: AtomGroup,
        ag3: AtomGroup,
        ag4: AtomGroup,
        use_mic: bool = False,
    ):
        n1 = len(ag1)
        n2 = len(ag2)
        n3 = len(ag3)
        n4 = len(ag4)
        if not n1 == n2 == n3 == n4:
            raise MDStuffError(
                f"the number of atoms in groups 1 ({n1}), 2 ({n2}), 3 ({n3}) and 4 ({n4}) do not match"
            )
        elif n1 == 0:
            raise MDStuffError(f"all atom groups are empty")

        super().__init__(ag1.universe)

        self.d1 = Distance(ag1=ag2, ag2=ag1, use_mic=use_mic)
        self.d2 = Distance(ag1=ag3, ag2=ag4, use_mic=use_mic)
        self.d3 = Distance(ag1=ag2, ag2=ag3, use_mic=use_mic)

        self._shape = (n1,)

    def __call__(self, *args, **kwargs):
        d1 = self.d1()
        d2 = self.d2()
        d3 = self.d3()
        d1xd3 = np.cross(d1, d3)
        d2xd3 = np.cross(d2, d3)
        d1xd3_norm = np.linalg.norm(d1xd3, axis=1)
        d2xd3_norm = np.linalg.norm(d2xd3, axis=1)
        a = np.arccos(np.sum(d1xd3 * d2xd3, axis=1) / (d1xd3_norm * d2xd3_norm))
        a *= 180.0 / np.pi
        return a


class CompoundDihedral(StructureFunction):
    def __init__(self, ag_list: List[AtomGroup]):
        for ag in ag_list:
            if len(ag) != 4:
                raise MDStuffError(f"all atom groups must contain exactly four atoms")

        super().__init__(ag_list[0].universe)

        self.ag_list = ag_list

        self._shape = (len(ag_list),)

    def __call__(self, *args, **kwargs):
        d1 = [ag.positions[0] - ag.positions[1] for ag in self.ag_list]
        d2 = [ag.positions[3] - ag.positions[2] for ag in self.ag_list]
        d3 = [ag.positions[2] - ag.positions[1] for ag in self.ag_list]
        d1xd3 = np.cross(d1, d3)
        d2xd3 = np.cross(d2, d3)
        d1xd3_norm = np.linalg.norm(d1xd3, axis=1)
        d2xd3_norm = np.linalg.norm(d2xd3, axis=1)
        a = np.arccos(np.sum(d1xd3 * d2xd3, axis=1) / (d1xd3_norm * d2xd3_norm))
        a *= 180.0 / np.pi
        return a


class CompoundDistance(StructureFunction):
    def __init__(
        self,
        ag_list1: List[AtomGroup],
        ag_list2: List[AtomGroup],
        use_mic: bool = True,
    ):
        n1 = len(ag_list1)
        n2 = len(ag_list2)
        if n1 != n2:
            raise MDStuffError(
                f"the number of compounds in ag_list1 ({n1}) does not match "
                f"the number of compounds in ag_list2 ({n2})"
            )
        if n1 == 0:
            raise MDStuffError(f"the atom group lists are empty")

        super().__init__(ag_list1[0].universe)

        self.ag_list1 = ag_list1
        self.ag_list2 = ag_list2
        self.use_mic = use_mic

        self._shape = (n1, 3)

    def __call__(self, *args, **kwargs):
        x1 = np.array([ag.center_of_mass() for ag in self.ag_list1])
        x2 = np.array([ag.center_of_mass() for ag in self.ag_list2])
        d = x2 - x1
        if self.use_mic:
            apply_mic(d, self.universe.dimensions[:3])
        return d


class CompoundPosition(StructureFunction):
    def __init__(
        self, ag_list: List[AtomGroup], use_pbc: bool = False,
    ):
        n = len(ag_list)
        if n == 0:
            raise MDStuffError(f"the atom group list is empty")

        super().__init__(ag_list[0].universe)

        self.ag_list = ag_list
        self.use_pbc = use_pbc

        self._shape = (n, 3)

    def __call__(self, *args, **kwargs):
        return np.array([ag.center_of_mass(pbc=self.use_pbc) for ag in self.ag_list])


class Dipole(StructureFunction):
    def __init__(self, ag_list: List[AtomGroup]):
        super().__init__(ag_list[0].universe)

        self.ag_list = ag_list

        self._shape = (len(ag_list), 3)

    def __call__(self, *args, **kwargs):
        x = np.array([ag.positions - ag.center_of_mass() for ag in self.ag_list])
        q = np.array([ag.charges for ag in self.ag_list])
        mu = np.sum(q[:, :, None] * x, axis=1)
        return mu


class CompoundAxis(StructureFunction):
    def __init__(self, ag_list: List[AtomGroup], n: int = 0):

        if n not in [0, 1, 2]:
            raise MDStuffError(f"n must be 0, 1, or 2")
        self.n = n

        n = len(ag_list)
        if n == 0:
            raise MDStuffError(f"the atom group list is empty")

        super().__init__(ag_list[0].universe)

        self.ag_list = ag_list

        self._shape = (n, 3)

    def __call__(self, *args, **kwargs):
        axes = np.array([ag.principal_axes()[self.n] for ag in self.ag_list])
        return axes


class Moments(StructureFunction):
    def __init__(self, ag_list: List[AtomGroup], n: int = 0):
        if n not in [0, 1, 2]:
            raise MDStuffError(f"n must be 0, 1, or 2")
        self.n = n

        n = len(ag_list)
        if n == 0:
            raise MDStuffError(f"the atom group list is empty")

        super().__init__(ag_list[0].universe)

        self.ag_list = ag_list

        self._shape = (n,)

    def __call__(self, *args, **kwargs):
        return np.array(
            [
                np.sort(np.linalg.eigh(ag.moment_of_inertia())[0])[2 - self.n]
                for ag in self.ag_list
            ]
        )
