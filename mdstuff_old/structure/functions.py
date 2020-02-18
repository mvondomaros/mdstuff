# import warnings
# from typing import Any, Tuple
#
# import MDAnalysis
# import numpy as np
#
# from .helpers import apply_mic, select_compounds
# from ..core.errors import MDStuffError
# from ..core.helpers import to_list
# from ..core.types import Selection
# from ..core.universe import Universe
#
#
# class StructureFunction:
#     """Base structure function class."""
#
#     def __init__(self) -> None:
#         """Initializations that do not depend on the universe being set"""
#         self.universe = None
#
#     def _late_init(self) -> None:
#         """Initializations that depend on the universe being set."""
#         pass
#
#     def set_universe(self, universe: Universe) -> None:
#         """
#         :param universe: an MDStuff universe
#         """
#         self.universe = universe
#         self._late_init()
#
#     def __call__(self, *args, **kwargs) -> Any:
#         pass
#
#     @property
#     def shape(self) -> Tuple:
#         """Return the shape of function values."""
#         return ()
#
#
# class AtomAxisPos(StructureFunction):
#     """
#     Computes the locations of atoms on an coordinate system axis.
#     """
#
#     def __init__(
#         self,
#         selection: Selection,
#         dimension: int = 2,
#         use_pbc: bool = False,
#         update: bool = False,
#     ):
#         """
#         :param selection: the selection string
#         :param dimension: optional, the dimension of the axis
#         :param use_pbc: optional, whether to use periodic boundary conditions
#         :param update: optional, whether to update the selections in each frame
#         """
#         super().__init__()
#
#         self.selection = selection
#         if dimension not in [0, 1, 2]:
#             raise MDStuffError(f"invalid dimension: {dimension}")
#         self.dimension = dimension
#         self.use_pbc = use_pbc
#         self.update = update
#
#     def _late_init(self) -> None:
#         self.ag = self.universe.select_atoms(
#             *to_list(self.selection), updating=self.update
#         )
#         if self.use_pbc:
#             self.box = self.universe.dimensions
#         else:
#             self.box = None
#
#     def __call__(self) -> np.ndarray:
#         x = self.ag.positions[:, self.dimension]
#         if self.use_pbc:
#             apply_mic(x, self.box[self.dimension])
#         return x
#
#
# class AtomCharge(StructureFunction):
#     """
#     Returns the charges of atoms.
#     """
#
#     def __init__(
#         self, selection: Selection, update: bool = False,
#     ):
#         """
#         :param selection: the selection string
#         :param update: optional, whether to update the selections in each frame
#         """
#         super().__init__()
#
#         self.selection = selection
#         self.update = update
#
#     def _late_init(self) -> None:
#         self.ag = self.universe.select_atoms(
#             *to_list(self.selection), updating=self.update
#         )
#
#     def __call__(self) -> np.ndarray:
#         return self.ag.charges
#
#
# class AtomAtomDist(StructureFunction):
#     """
#     Computes the distances r_12=||r_2-r_1|| between pairs of atoms constructed from two atom selections 1 and 2.
#     """
#
#     def __init__(
#         self,
#         selection1: Selection,
#         selection2: Selection,
#         mode="product",
#         use_mic: bool = True,
#         update: bool = False,
#     ) -> None:
#         """
#         :param selection1: the selection string for the first atom group
#         :param selection2: the selection string for the second atom group
#         :param mode: optional, "product" or "zip"
#             zip atoms together or take unique pairs from the cartesian product of both groups
#         :param use_mic: optional, whether to use the minimum image convention
#         :param update: optional, whether to update the selections in each frame
#         """
#         super().__init__()
#
#         self.selection1 = selection1
#         self.selection2 = selection2
#         self.update = update
#         if mode not in ["product", "zip"]:
#             raise MDStuffError(f"invalid mode: {mode}")
#         self.mode = mode
#         self.use_mic = use_mic
#
#     def _late_init(self) -> None:
#         self.ag1 = self.universe.select_atoms(
#             *to_list(self.selection1), updating=self.update
#         )
#         self.ag2 = self.universe.select_atoms(
#             *to_list(self.selection2), updating=self.update
#         )
#         if self.mode == "zip":
#             n1 = len(self.ag1)
#             n2 = len(self.ag2)
#             if n1 != n2:
#                 raise MDStuffError(
#                     f"the number of atoms in selection1 and selection2 do not match"
#                 )
#
#         if self.use_mic:
#             self.box = self.universe.dimensions
#         else:
#             self.box = None
#
#     def __call__(self) -> np.ndarray:
#         r1 = self.ag1.positions
#         r2 = self.ag2.positions
#         if self.mode == "product":
#             if self.selection1 == self.selection2:
#                 d = MDAnalysis.lib.distances.self_distance_array(
#                     reference=r1, box=self.box, backend="OpenMP",
#                 )
#             else:
#                 d = MDAnalysis.lib.distances.distance_array(
#                     reference=r2, configuration=r1, box=self.box, backend="OpenMP",
#                 ).ravel()
#         elif self.mode == "zip":
#             d = r2 - r1
#             if self.use_mic:
#                 apply_mic(d, self.box[:3])
#             d = np.linalg.norm(d, axis=1)
#         else:
#             raise NotImplementedError(f"{self.mode=}")
#         return d
#
#     @property
#     def shape(self) -> Tuple:
#         n = len(self.ag1)
#         if self.ag2 is None:
#             shape = (n * (n - 1) // 2,)
#         else:
#             m = len(self.ag2)
#             shape = (n * m,)
#         return shape
#
#
# class AtomAtomDistVect(AtomAtomDist):
#     """
#     Computes the distance vectors r_12=r_2-r_1 between pairs of atoms constructed from two atom selections 1 and 2.
#     """
#
#     def __call__(self) -> np.ndarray:
#         r1 = self.ag1.positions
#         r2 = self.ag2.positions
#         if self.mode == "product":
#             d = r2[None, :] - r1[:, None]
#             if self.selection1 == self.selection2:
#                 d = d[np.triu_indices(d.shape[0], k=1)]
#             d = d.reshape(-1, 3)
#             if self.use_mic:
#                 apply_mic(d, self.box[:3])
#         elif self.mode == "zip":
#             d = r2 - r1
#             if self.use_mic:
#                 apply_mic(d, self.box[:3])
#         else:
#             raise NotImplementedError(f"{self.mode=}")
#         return d
#
#     @property
#     def shape(self) -> Tuple:
#         n = len(self.ag1)
#         if self.ag2 is None:
#             shape = (n * (n - 1) // 2,)
#         else:
#             m = len(self.ag2)
#             shape = (n * m,)
#         return shape
#
#
# class Angle(StructureFunction):
#     """Computes angles between atom triplets (1--2--3)."""
#
#     def __init__(
#         self,
#         selection1: Selection,
#         selection2: Selection,
#         selection3: Selection,
#         use_mic: bool = True,
#         update: bool = False,
#     ) -> None:
#         """
#         :param selection1: the selection string for the first atom group
#         :param selection2: the selection string for the second atom group
#         :param selection2: the selection string for the second atom group
#         :param use_mic: optional, whether to use the minimum image convention
#         :param update: optional, whether to update the selections in each frame
#         """
#         super().__init__()
#
#         self.selection1 = selection1
#         self.selection2 = selection2
#         self.selection3 = selection3
#         self.update = update
#         self.use_mic = use_mic
#
#     def _late_init(self) -> None:
#         self.ag1 = self.universe.select_atoms(
#             *to_list(self.selection1), updating=self.update
#         )
#         self.ag2 = self.universe.select_atoms(
#             *to_list(self.selection2), updating=self.update
#         )
#         self.ag3 = self.universe.select_atoms(
#             *to_list(self.selection3), updating=self.update
#         )
#         n1 = len(self.ag1)
#         n2 = len(self.ag2)
#         n3 = len(self.ag3)
#         if not (n1 == n2 == n3):
#             warnings.warn(
#                 f"The numbers of atoms do not match between selection1 (n={n1}), selection2 (n={n2}), and "
#                 f"selection3 (n={n3}).\n"
#                 f"This might be intended, so I'm going to repeat the selections until they match.\n"
#                 f"Be very careful with this feature. Below is the new angle list."
#             )
#             # Equalize groups 1 and 2, first.
#             n12 = np.lcm(n1, n2)
#             self.ag1 = sum((n12 // n1) * [self.ag1])
#             self.ag2 = sum((n12 // n2) * [self.ag2])
#             # Now equalize all groups.
#             n123 = np.lcm(n12, n3)
#             if n12 != n123:
#                 self.ag1 = sum((n123 // n12) * [self.ag1])
#                 self.ag2 = sum((n123 // n12) * [self.ag2])
#             self.ag3 = sum((n123 // n3) * [self.ag3])
#             for a1, a2, a3 in zip(self.ag1, self.ag2, self.ag3):
#                 print(
#                     f"{a1.name} (resid {a1.resid}), {a2.name} (resid {a2.resid}), {a3.name} (resid {a3.resid})"
#                 )
#
#         if self.use_mic:
#             self.box = self.universe.dimensions
#         else:
#             self.box = None
#
#     def __call__(self) -> np.ndarray:
#         r1 = self.ag1.positions
#         r2 = self.ag2.positions
#         r3 = self.ag3.positions
#         return (
#             MDAnalysis.lib.distances.calc_angles(
#                 r1, r2, r3, box=self.box, backend="OpenMP"
#             )
#             * 180.0
#             / np.pi
#         )
#
#     @property
#     def shape(self) -> Tuple:
#         return (len(self.ag1),)
#
#
# class DipoleMoment(StructureFunction):
#     """Computes the dipole moments of selected compounds."""
#
#     def __init__(self, selection: Selection, compound: str,) -> None:
#         """
#         :param selection: the selection string
#         :param compound: a compound specifier
#         """
#         super().__init__()
#
#         self.selection = selection
#         self.compound = compound
#         self.ag_list = []
#
#     def _late_init(self) -> None:
#         self.ag_list = select_compounds(
#             universe=self.universe, selection=self.selection, compound=self.compound
#         )
#
#     def _dipole_vectors(self):
#         return np.array(
#             [
#                 np.sum(
#                     (ag.positions - ag.center_of_mass()) * ag.charges[:, None], axis=0
#                 )
#                 for ag in self.ag_list
#             ]
#         )
#
#     def __call__(self) -> np.ndarray:
#         return np.linalg.norm(self._dipole_vectors(), axis=1)
#
#
# class DipoleMomentOrientation(DipoleMoment):
#     """
#     Computes the orientations of he dipole moments of selected compounds with respect to an axis.
#     Returns the cosine of the angles between dipole moments and axis.
#     """
#
#     def __init__(self, selection: Selection, compound: str, dimension: int = 2) -> None:
#         """
#         :param selection: the selection string
#         :param compound: a compound specifier
#         :param dimension: optional, the dimension of the chosen axis
#         """
#         super().__init__(selection=selection, compound=compound)
#
#         if dimension not in [0, 1, 2]:
#             raise MDStuffError(f"invalid dimension: {dimension}")
#         self.dimension = dimension
#
#     def __call__(self) -> np.ndarray:
#         mu = self._dipole_vectors()
#         return mu[:, self.dimension] / np.linalg.norm(mu, axis=1)
