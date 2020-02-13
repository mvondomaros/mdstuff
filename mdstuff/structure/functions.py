from typing import Any, Tuple

import MDAnalysis
import numpy as np

from .helpers import apply_mic_1d
from ..core.errors import MDStuffError
from ..core.helpers import to_list
from ..core.types import Selection
from ..core.universe import Universe


class StructureFunction:
    """Base structure function class."""

    def __init__(self) -> None:
        """Initializations that do not depend on the universe being set"""
        self.universe = None

    def _late_init(self) -> None:
        """Initializations that depend on the universe being set."""
        pass

    def set_universe(self, universe: Universe) -> None:
        """
        :param universe: an MDStuff universe
        """
        self.universe = universe
        self._late_init()

    def __call__(self, *args, **kwargs) -> Any:
        pass

    @property
    def shape(self) -> Tuple:
        """Return the shape of function values."""
        return ()


class IntraCompoundDistance(StructureFunction):
    """
    Computes distances withing compounds.

    Depending on what compound is specified, these can be intra-molecular/intra-residue atom-atom distances, or
    intra-segment/intra-fragment residue-residue distances.
    """

    def __init__(
        self, selection1: Selection, selection2: Selection, compound: str,
    ) -> None:
        """
        :param selection1: a selection string
        :param selection2: a selection string
        :param compound: a compound specifier (except for "group")
        """
        super().__init__()

        self.selection1 = to_list(selection1)
        self.selection2 = to_list(selection2)
        self.compound = compound

        # Determine the minor compound.
        if compound == "residues" or compound == "molecules":
            # Assuming user wants to calculate atom-wise distances.
            self.minor_compound = None
        elif compound == "segments" or compound == "fragments":
            # Assuming user wants to calculate residue-wise distances.
            self.minor_compound = "residues"
        else:
            raise MDStuffError(f"invalid compound specifier: {compound=}")

        self.ag1_list = []
        self.ag2_list = []

    def _late_init(self) -> None:
        for c in getattr(self.universe, self.compound):
            ag1 = c.atoms.select_atoms(*self.selection1)
            ag2 = c.atoms.select_atoms(*self.selection2)
            if len(ag1) != len(ag2):
                raise MDStuffError(
                    f"atom number mismatch for selection1={self.selection1} and selection2={self.selection2} in "
                    f"{self.compound[:-1]} {c.ix}"
                )
            elif len(ag1) != 0:
                self.ag1_list.append(ag1)
                self.ag2_list.append(ag2)

    def __call__(self) -> np.ndarray:
        if self.minor_compound is None:
            x1 = np.concatenate([ag.positions for ag in self.ag1_list])
            x2 = np.concatenate([ag.positions for ag in self.ag2_list])
        else:
            x1 = np.concatenate(
                [
                    ag.center_of_mass(compound=self.minor_compound)
                    for ag in self.ag1_list
                ]
            )
            x2 = np.concatenate(
                [
                    ag.center_of_mass(compound=self.minor_compound)
                    for ag in self.ag2_list
                ]
            )
        d = np.linalg.norm(x1 - x2, axis=1)
        return d

    @property
    def shape(self) -> Tuple:
        n = self.universe.nr_compounds(
            compound=self.compound, selection=self.selection1
        )
        m = self.universe.nr_compounds(
            compound=self.compound, selection=self.selection2
        )
        return n, m


class InterCompoundDistance(StructureFunction):
    """Computes distances between atoms and/or between compounds."""

    def __init__(
        self,
        selection: Selection,
        other_selection: Selection = None,
        compound: str = None,
        other_compound: str = None,
        use_mic: bool = True,
    ) -> None:
        """
        :param selection: the first selection string
        :param other_selection: optional, the second selection string
        :param compound: optional, a compound specifier for the first selection
        :param other_compound: optional, a compound specifier for the second selection
        :param use_mic: optional, whether to use the minimum image convention
        """
        super().__init__()

        self.selection = to_list(selection)
        self.other_selection = to_list(other_selection)
        self.ag = None
        self.other_ag = None
        self.compound = compound
        self.other_compound = other_compound
        self.use_mic = use_mic

    def _late_init(self) -> None:
        self.ag = self.universe.select_atoms(*self.selection)
        if self.other_selection is not None:
            self.other_ag = self.universe.select_atoms(*self.other_selection)
        if self.use_mic:
            self.box = self.universe.dimensions
        else:
            self.box = None

    def __call__(self) -> np.ndarray:
        if self.compound is None:
            x = self.ag.positions
        else:
            x = self.ag.center_of_mass(compound=self.compound)
        if self.other_ag is not None:
            if self.other_compound is None:
                y = self.other_ag.positions
            elif self.other_compound == "group":
                y = self.other_ag.ag2.center_of_mass(
                    compound=self.other_compound
                ).reshape((1, -1))
            else:
                y = self.other_ag.center_of_mass(compound=self.other_compound)
            d = MDAnalysis.lib.distances.distance_array(
                reference=y, configuration=x, box=self.box, backend="OpenMP",
            ).reshape(-1)
        else:
            d = MDAnalysis.lib.distances.self_distance_array(
                reference=x, box=self.box, backend="OpenMP",
            )
        return d

    @property
    def shape(self) -> Tuple:
        n = self.universe.nr_compounds(compound=self.compound, selection=self.selection)
        if self.other_selection is None:
            return (n * (n - 1) // 2,)
        else:
            m = self.universe.nr_compounds(
                compound=self.other_compound, selection=self.other_selection
            )
            return (n * m,)


class InterCompoundDistanceProjection(InterCompoundDistance):
    """Computes the projection of an InterCompoundDistance onto an axis."""

    def __init__(
        self,
        selection: Selection,
        other_selection: Selection = None,
        compound: str = None,
        other_compound: str = None,
        dimension: int = 2,
        use_mic: bool = True,
    ) -> None:
        """
        :param selection: the first selection string
        :param other_selection: optional, the second selection string
        :param compound: optional, a compound specifier for the first selection
        :param other_compound: optional, a compound specifier for the second selection
        :param dimension: optional, the dimension of the projection axis
        :param use_mic: optional, whether to use the minimum image convention
        """
        super().__init__(
            selection=selection,
            other_selection=other_selection,
            compound=compound,
            other_compound=other_compound,
            use_mic=use_mic,
        )
        if 0 <= dimension <= 2:
            self.dimension = dimension
        else:
            raise MDStuffError(f"invalid dimension: {dimension=}")

    def __call__(self) -> np.ndarray:
        if self.compound is None:
            x = self.ag.positions[:, self.dimension]
        else:
            x = self.ag.center_of_mass(compound=self.compound)[:, self.dimension]
        if self.other_ag is not None:
            if self.other_compound is None:
                y = self.other_ag.positions[:, self.dimension]
            elif self.other_compound == "group":
                y = self.other_ag.center_of_mass(compound=self.other_compound)[
                    self.dimension
                ]
            else:
                y = self.other_ag.center_of_mass(compound=self.other_compound)[
                    :, self.dimension
                ]
            d = y[:, None] - x[None, :]
        else:
            # Behold the numpy magic!
            d = x[:, None] - x[None, :]
            d = d[np.triu_indices_from(d, k=1)]
        if self.box is not None:
            apply_mic_1d(d, self.box[self.dimension])
        return d


class AxisPosition(StructureFunction):
    """Computes the positions of atoms along an axis."""

    def __init__(
        self, selection: Selection, dimension: int = 2, use_pbc: bool = True,
    ) -> None:
        """
        :param selection: the selection string
        :param dimension: optional, the dimension of the projection axis
        :param use_pbc: optional, whether to use periodic boundary conditions
        """
        super().__init__()

        self.selection = to_list(selection)
        self.ag = None
        self.use_pbc = use_pbc

        if 0 <= dimension <= 2:
            self.dimension = dimension
        else:
            raise MDStuffError(f"invalid dimension: {dimension=}")

    def _late_init(self) -> None:
        self.ag = self.universe.select_atoms(*self.selection)

    def __call__(self) -> np.ndarray:
        return self.ag.positions[:, self.dimension]

    @property
    def shape(self) -> Tuple:
        return (len(self.ag),)


class CompoundAxisPosition(AxisPosition):
    """Computes the positions of compounds along an axis."""

    def __init__(
        self,
        selection: Selection,
        compound: str = None,
        dimension: int = 2,
        use_pbc: bool = True,
    ) -> None:
        """
        :param selection: the selection string
        :param compound: optional, a compound specifier
        :param dimension: optional, the dimension of the projection axis
        :param use_pbc: optional, whether to use periodic boundary conditions
        """
        super().__init__(selection=selection, dimension=dimension, use_pbc=use_pbc)

        self.compound = compound

    def __call__(self) -> np.ndarray:
        return self.ag.center_of_mass(compound=self.compound, pbc=self.use_pbc)[
            :, self.dimension
        ]

    @property
    def shape(self) -> Tuple:
        return (
            self.universe.nr_compounds(
                compound=self.compound, selection=self.selection
            ),
        )


class Charge(StructureFunction):
    """Returns the charges of atoms."""

    def __init__(self, selection: Selection) -> None:
        """
        :param selection: the selection string
        """
        super().__init__()

        self.selection = to_list(selection)
        self.ag = None

    def _late_init(self) -> None:
        self.ag = self.universe.select_atoms(*self.selection)

    def __call__(self) -> np.ndarray:
        return self.ag.charges

    @property
    def shape(self) -> Tuple:
        return (len(self.ag),)


class Dipole(StructureFunction):
    """Computes the dipole of a compound."""

    def __init__(self, selection: Selection, compound: str) -> None:
        """
        :param selection: the selection string
        :param compound: a compound specifier
        """
        super().__init__()

        self.selection = to_list(selection)
        self.compound = compound
        self.ag_list = []

    def _late_init(self) -> None:
        for c in getattr(self.universe, self.compound):
            ag = c.atoms.select_atoms(*self.selection)
            if len(ag) != 0:
                self.ag_list.append(ag)

    def __call__(self) -> np.ndarray:
        dipoles = []
        for ag in self.ag_list:
            mu = np.sum(((ag.positions - ag.center_of_mass()) * ag.charges), axis=0,)
            dipoles.append(mu)
        return np.array(dipoles)

    @property
    def shape(self) -> Tuple:
        return self.universe.nr_compounds(
            compound=self.compound, selection=self.selection
        )


class DipoleOrientation(Dipole):
    """Computes the dipole orientation of a compound."""

    def __init__(self, selection: Selection, compound: str, dimension: int = 2) -> None:
        """
        :param selection: the selection string
        :param compound: a compound specifier
        """
        super().__init__(selection=selection, compound=compound)

        if 0 <= dimension <= 2:
            self.dimension = dimension
        else:
            raise MDStuffError(f"invalid dimension: {dimension=}")

    def __call__(self) -> np.ndarray:
        dipoles = super().__call__()
        dipole_moments = np.linalg.norm(dipoles, axis=1)
        return dipoles[:, self.dimension] / dipole_moments


#
# class DipoleOrientation(StructureFunction):
#     """Computes the orientation of a dipole moment."""
#
#     def __init__(
#         self,
#         universe: MDAnalysis.Universe,
#         selection: Selection,
#         compound: str,
#         dimension: int = 2,
#     ) -> None:
#         """
#         :param universe: an MDAnalysis Universe
#         :param selection: a selection string
#         :param compound: an MDAnalysis compound specifier
#         :param dimension: optional, the dimension of the projection axis
#         """
#         super().__init__(universe=universe)
#
#         selection = to_list(selection)
#         self.ag_list = []
#         if not hasattr(self.universe, compound):
#             raise MDStuffError(f"invalid compound specifier: {compound=}")
#         for c in getattr(self.universe, compound):
#             ag = c.atoms.select_atoms(*selection)
#             if len(ag) != 0:
#                 self.ag_list.append(ag)
#
#         if 0 <= dimension <= 2:
#             self.dimension = dimension
#         else:
#             raise MDStuffError(f"invalid dimension: {dimension=}")
#
#     def __call__(self) -> np.ndarray:
#         """
#         Evaluate the function.
#
#         :return: a numpy array
#         """
#         mu = [
#             np.sum((ag.positions - ag.center_of_mass()) * ag.charges, axis=0)
#             for ag in self.ag_list
#         ]
#         mu = np.array([m[self.dimension] / np.linalg.norm(m) for m in mu])
#         return mu
