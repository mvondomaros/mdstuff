from __future__ import annotations

import MDAnalysis
import abc
import itertools
import numpy as np
import tqdm
from MDAnalysis.core.groups import AtomGroup
from typing import Any, Tuple, List

from .errors import MDStuffError, UniverseError, ParameterValueError


class Analysis:
    """
    Base analysis class.
    """

    def __init__(self, universe: Universe):
        """Set the universe."""
        self.universe = universe

    @abc.abstractmethod
    def update(self):
        """Update the analysis."""
        pass

    @abc.abstractmethod
    def finalize(self):
        """Finalize the analysis."""
        pass

    @abc.abstractmethod
    def get(self, *args, **kwargs) -> Any:
        """
        Return the results.

        :return: the results
        """
        pass


class OneTimeAnalysis(Analysis, abc.ABC):
    """
    Base class for one-time analyses: analyses that do not need to be updated continuously, but that do
    all their work in final(). The update() method won't be called for these analyses.
    """

    def update(self):
        """Should never be called."""
        raise NotImplementedError


class Universe(MDAnalysis.Universe):
    """
    Base universe class.

    This is an MDAnalysis universe which keeps track of analyses to perform. Unlike in MDAnalysis, this class is a
    Singleton.
    """

    # Singleton instance reference.
    instance = None

    # A list of valid compound specifiers.
    _VALID_COMPOUNDS = ["residues", "molecules", "fragments", "segments"]

    def __init__(self, *args, **kwargs):
        # Check if there's already another universe.
        if Universe.instance is not None:
            raise UniverseError

        Universe.instance = self
        super().__init__(*args, **kwargs)

        # The list of analyses to perform.
        self.analyses = []

    # Needed because of ABC requirements, but not implemented.
    def __getstate__(self, *args, **kwargs):
        raise NotImplementedError

    # Needed because of ABC requirements, but not implemented.
    def __setstate__(self, *args, **kwargs):
        raise NotImplementedError

    def add_analysis(self, analysis: Analysis):
        """
        Add an analysis.

        :param analysis: the analysis
        """
        self.analyses.append(analysis)

    def run_analyses(self, start: int = None, stop: int = None, step: int = None):
        """
        Run all analyses.

        :param start: optional, the first time step
        :param stop: optional, the last time step
        :param step: optional, the time step increment
        """
        # Nothing to do.
        if len(self.analyses) == 0:
            return

        # Run update() for each analysis, but only if there's actually a running analysis.
        is_ota = [isinstance(analysis, OneTimeAnalysis) for analysis in self.analyses]
        if not np.all(is_ota):
            for _ in tqdm.tqdm(
                self.trajectory[start:stop:step], desc="main trajectory loop"
            ):
                for analysis in self.analyses:
                    analysis.update()

        # Run finalize() for each analysis.
        for analysis in tqdm.tqdm(self.analyses, desc="analysis finalization loop"):
            analysis.finalize()

        # Clear the list of analyses.
        self.analyses = []

    def select_atom_pairs(
        self,
        selection1: str,
        selection2: str,
        mode: str = "zip",
        compound: str = "residues",
    ) -> Tuple[AtomGroup, AtomGroup]:
        """
        Select unique pairs of atoms. Returns two atom lists of equal length, corresponding to the first and second
        elements of the pairs, respectively.

        :param selection1: the first selection string
        :param selection2: the second selection string
        :param mode: optional, the mode
            "zip": zip over atoms in selection1 and selection2
            "product": form the cartesian product between atoms in selection1 and selection2
            "within": form the cartesian product between atoms in selection1 and selection2 that are within the same
                compound
            "between": form the cartesian product between atoms in selection1 and selection2 that are *not* within the
                same compound
        :param compound: optional, the compound specifier for mode="between" or mode="within"
        :return: two atom groups
        """
        # Check the compound specifier.
        if compound not in self._VALID_COMPOUNDS:
            raise ParameterValueError(
                name="compound", value=compound, allowed_values=self._VALID_COMPOUNDS
            )

        # Check the mode.
        _VALID_MODES = ["zip", "product", "within", "between"]
        if mode not in _VALID_MODES:
            raise ParameterValueError(
                name="mode", value=mode, allowed_values=_VALID_MODES
            )

        # In the following, we construct two collections: a list of pair indices (ordered), and a set of pair indices
        # (unordered, unique). Only pairs that are not in the set will be added to the list. In the end, the atom groups
        # will be constructed from the indices in the list.
        index_set = set()
        index_list = list()

        if mode == "zip":
            # Here, we zip over pairs of atoms in ag1 and ag2.
            ag1 = self.select_atoms(selection1)
            ag2 = self.select_atoms(selection2)
            if (n1 := len(ag1)) != (n2 := len(ag2)):
                raise MDStuffError(
                    f"the number of atoms in selection1 ({n1}) does not match "
                    f"the number of atoms in selection2 ({n2})"
                )
            for i1, i2 in zip(ag1.ix, ag2.ix):
                if (i1, i2) not in index_set:
                    index_set.add((i1, i2))
                    index_list.append((i1, i2))
        elif mode == "product":
            # Here, we form the cartesian product of atoms in ag1 and ag2.
            ag1 = self.select_atoms(selection1)
            ag2 = self.select_atoms(selection2)
            for i1, i2 in itertools.product(ag1.ix, ag2.ix):
                if (i1, i2) not in index_set:
                    index_set.add((i1, i2))
                    index_list.append((i1, i2))
        elif mode == "within":
            # Here, we loop over all selected compounds, select the respective atoms, and form the cartesian product.
            for c in getattr(self, compound):
                ag1 = c.atoms.select_atoms(selection1)
                ag2 = c.atoms.select_atoms(selection2)
                if len(ag1) == 0 or len(ag2) == 0:
                    # Cartesian product will be empty, no need to continue.
                    continue
                for i1, i2 in itertools.product(ag1.ix, ag2.ix):
                    if (i1, i2) not in index_set:
                        index_set.add((i1, i2))
                        index_list.append((i1, i2))
        elif mode == "between":
            # Here, we loop over all pairs of unequal compounds, select the respective atoms, and form the cartesian
            # product.
            compounds = getattr(self, compound)
            for c1 in compounds:
                ag1 = c1.atoms.select_atoms(selection1)
                if len(ag1) == 0:
                    # Cartesian product will be empty, no need to continue.
                    continue
                for c2 in compounds:
                    if c1 == c2:
                        # Same compound. No need to continue.
                        continue
                    ag2 = c2.atoms.select_atoms(selection2)
                    if len(ag2) == 0:
                        # Cartesian product will be empty, no need to continue.
                        continue
                    for i1, i2 in itertools.product(ag1.ix, ag2.ix):
                        if (i1, i2) not in index_set:
                            index_set.add((i1, i2))
                            index_list.append((i1, i2))
        else:
            raise NotImplementedError

        i1, i2 = np.transpose(index_list)
        return AtomGroup(i1, self), AtomGroup(i2, self)

    def select_atom_triplets(
        self, ag1: AtomGroup, ag2: AtomGroup, selection: str, mode: str = "zip",
    ) -> Tuple[
        AtomGroup, AtomGroup, AtomGroup,
    ]:
        """
        Select unique triplets of atoms. Returns three atom lists of equal length, corresponding to the first, second,
        and third elements of the triplets, respectively.

        :param ag1: the first atom group (use select_pairs() to get the desired selection)
        :param ag2: the second atom group (use select_pairs() to get the desired selection)
        :param selection: the selection string for the third atom group
        :param mode: optional, the mode
            "zip": zip over atoms in ag1, ag2 and selection3
            "product": form the cartesian product between pairs of atoms in ag1 and ag2 and those atoms in selection3
        :return: two atom groups
        """
        # Check the mode.
        _VALID_MODES = ["zip", "product"]
        if mode not in _VALID_MODES:
            raise ParameterValueError(
                name="mode", value=mode, allowed_values=_VALID_MODES
            )

        # Check the lengths of ag1 and ag2.
        if (n1 := len(ag1)) != (n2 := len(ag2)):
            raise MDStuffError(
                f"the number of atoms in ag1 ({n1}) does not match the number of atoms in ag2 ({n2})"
            )

        # In the following, we construct two collections: a list of triplet indices (ordered), and a set of triplet
        # indices (unordered, unique). Only triplets that are not in the set will be added to the list. In the end,
        # the atom groups will be constructed from the indices in the list.
        index_set = set()
        index_list = list()

        if mode == "zip":
            # Here, we zip over triplets of atoms in ag1, ag2, and ag3.
            ag3 = self.select_atoms(selection)
            if n1 != (n3 := len(ag3)):
                raise MDStuffError(
                    f"the number of atoms in ag1 and ag2 ({n1}) does not match "
                    f"the number of selected atoms ({n3})"
                )
            for i1, i2, i3 in zip(ag1.ix, ag2.ix, ag3.ix):
                if (i1, i2, i3) not in index_set:
                    index_set.add((i1, i2, i3))
                    index_list.append((i1, i2, i3))
        elif mode == "product":
            # Here, we form the cartesian product between pairs in ag1 and ag2, and atoms ag3.
            ag3 = self.select_atoms(selection)
            for (i1, i2), i3 in itertools.product(zip(ag1.ix, ag2.ix), ag3.ix):
                if (i1, i2, i3) not in index_set:
                    index_set.add((i1, i2, i3))
                    index_list.append((i1, i2, i3))
        else:
            raise NotImplementedError
        i1, i2, i3 = np.transpose(index_list)
        return AtomGroup(i1, self), AtomGroup(i2, self), AtomGroup(i3, self)

    def select_compounds(
        self, selection: str, compound: str = "residues"
    ) -> List[AtomGroup]:
        """
        Select compounds of atoms.

        :param selection: the selection string
        :param compound: optional, the compound specifier
        :return: a list of atom groups
        """
        ag_list = []
        if compound == "group":
            ag = self.select_atoms(selection)
            if len(ag) != 0:
                ag_list.append(ag)
        elif compound in self._VALID_COMPOUNDS:
            for c in getattr(self, compound):
                ag = c.atoms.select_atoms(selection)
                if len(ag) != 0:
                    ag_list.append(ag)
        else:
            raise ParameterValueError(
                name="compound",
                value=compound,
                allowed_values=self._VALID_COMPOUNDS + ["group"],
            )

        return ag_list
