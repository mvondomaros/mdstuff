from __future__ import annotations

import abc
import itertools
import sys
from typing import Any, Tuple

import MDAnalysis
import numpy as np
import tqdm
from MDAnalysis.core.groups import AtomGroup

from .errors import MDStuffError


class Analysis:
    """
    Base analysis class.
    """

    MODE = "RA"

    def __init__(self, universe: Universe):
        """Set the universe."""
        self.universe = universe

    @abc.abstractmethod
    def update(self):
        """Update the analysis."""
        pass

    @abc.abstractmethod
    def finalize(self, start: int, stop: int, step: int):
        """Finalize the anaylis."""
        pass

    @abc.abstractmethod
    def get(self, *args, **kwargs) -> Any:
        """
        Return the results.

        :return: the results
        """
        pass


class Universe(MDAnalysis.Universe):
    """
    Base universe class.

    This is an MDAnalysis universe which keeps track of analyses to perform.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.analyses = []

    def __getstate__(self, *args, **kwargs):
        raise NotImplementedError

    def __setstate__(self, *args, **kwargs):
        raise NotImplementedError

    def add_analysis(self, analysis: Analysis):
        """
        Add an analysis.

        :param analysis: the analysis
        """
        if analysis.universe == self:
            self.analyses.append(analysis)
        else:
            raise MDStuffError(f"analysis works in a different universe")

    def run_analyses(self, start: int = 0, stop: int = -1, step: int = 1):
        """
        Run all analyses.

        :param start: optional, the first time step
        :param stop: optional, the last time step
        :param step: optional, the time step increment
        """
        # Check if we can skip the main loop, because all analyses are OTAs.
        modes = [analysis.MODE for analysis in self.analyses]
        if np.any(np.array(modes) == "RA"):
            for _ in tqdm.tqdm(
                self.trajectory[start:stop:step], desc="main trajectory loop"
            ):
                for analysis in self.analyses:
                    analysis.update()

        for analysis in tqdm.tqdm(self.analyses, desc="analysis finalization loop"):
            analysis.finalize(start=start, stop=stop, step=step)
        self.analyses = []

    def select_atom_pairs(
        self,
        selection1: str,
        selection2: str,
        mode: str = "zip",
        compound: str = "residues",
    ) -> Tuple[AtomGroup, AtomGroup]:
        """
        Select pairs of atoms. Returns two atom lists of equal length, corresponding to the first and second
        elements of the pairs, respectively.

        :param selection1: the first selection string
        :param selection2: the second selection string
        :param mode: optional, the mode
            "zip": zip over selected atoms
            "product": return unique pairs of atoms from the cartesian product of both selections
            "within": return unique pairs of atoms within the same compound from the cartesian product of both
                selections
            "between": return unique pairs of atoms between different compounds from the cartesian product of both
                selections
        :param compound: optional, the compound specifier for mode="between" or mode="within"
        :return:
        """
        if compound is not None and compound not in [
            "residues",
            "molecules",
            "fragments",
            "segments",
        ]:
            raise MDStuffError(f"invalid parameter value: {compound=}")
        if mode not in ["zip", "product", "within", "between"]:
            raise MDStuffError(f"invalid parameter value: {mode=}")

        # Check if there are any atoms selected at all.
        ag1 = self.select_atoms(selection1)
        ag2 = self.select_atoms(selection2)
        n1 = len(ag1)
        n2 = len(ag2)
        if n1 == 0:
            raise MDStuffError(f"selection1 is empty")
        if n2 == 0:
            raise MDStuffError(f"selection2 is empty")

        # Construct a set and an ordered list of pair indices.
        index_set = set()
        index_list = list()

        if mode == "zip":
            if n1 != n2:
                raise MDStuffError(
                    f"the number of atoms in selection1 ({n1}) does not match "
                    f"the number of atoms in selection2 ({n2})"
                )
            for i1, i2 in zip(ag1.ix, ag2.ix):
                if (i1, i2) not in index_set:
                    index_set.add((i1, i2))
                    index_list.append((i1, i2))
        elif mode == "product":
            for i1, i2 in itertools.product(ag1.ix, ag2.ix):
                if (i1, i2) not in index_set:
                    index_set.add((i1, i2))
                    index_list.append((i1, i2))
        elif mode == "within":
            for c in getattr(self, compound):
                ag1 = c.atoms.select_atoms(selection1)
                ag2 = c.atoms.select_atoms(selection2)
                n1 = len(ag1)
                n2 = len(ag2)
                if n1 == 0 or n2 == 0:
                    continue
                for i1, i2 in itertools.product(ag1.ix, ag2.ix):
                    if (i1, i2) not in index_set:
                        index_set.add((i1, i2))
                        index_list.append((i1, i2))
        elif mode == "between":
            compounds = getattr(self, compound)
            for c1 in compounds:
                ag1 = c1.atoms.select_atoms(selection1)
                n1 = len(ag1)
                if n1 == 0:
                    continue
                for c2 in compounds:
                    if c1 == c2:
                        continue
                    ag2 = c2.atoms.select_atoms(selection2)
                    n2 = len(ag2)
                    if n2 == 0:
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
        self,
        ag1: AtomGroup,
        ag2: AtomGroup,
        selection: str,
        mode: str = "zip",
        compound: str = "residues",
    ) -> Tuple[
        AtomGroup, AtomGroup, AtomGroup,
    ]:
        """
        Select triplets of atoms.
        """
        n1 = len(ag1)
        n2 = len(ag2)
        if n1 != n2:
            raise MDStuffError(
                f"the number of atoms in ag1 ({n1}) does not match the number of atoms in ag2 ({n2})"
            )
        if n1 == 0:
            raise MDStuffError(f"no atoms selected in ag1 and ag2")
        if compound is not None and compound not in [
            "residues",
            "molecules",
            "fragments",
            "segments",
        ]:
            raise MDStuffError(f"invalid parameter value: {compound=}")
        if mode not in ["zip", "product", "within", "between"]:
            raise MDStuffError(f"invalid parameter value: {mode=}")

        # Check if there are any atoms selected at all.
        ag3 = self.select_atoms(selection)
        n3 = len(ag3)
        if n3 == 0:
            raise MDStuffError(f"selection is empty")

        # Construct a set and an ordered list of pair indices.
        index_set = set()
        index_list = list()

        if mode == "zip":
            if n1 != n3:
                raise MDStuffError(
                    f"the number of atoms in ag1 and ag2 ({n1}) does not match "
                    f"the number of selected atoms ({n3})"
                )
            for i1, i2, i3 in zip(ag1.ix, ag2.ix, ag3.ix):
                if (i1, i2, i3) not in index_set:
                    index_set.add((i1, i2, i3))
                    index_list.append((i1, i2, i3))
        elif mode == "product":
            for (i1, i2), i3 in itertools.product(zip(ag1.ix, ag2.ix), ag3.ix):
                if (i1, i2, i3) not in index_set:
                    index_set.add((i1, i2, i3))
                    index_list.append((i1, i2, i3))
        elif mode == "within":
            raise NotImplementedError
        elif mode == "between":
            raise NotImplementedError
        else:
            raise NotImplementedError
        i1, i2, i3 = np.transpose(index_list)
        return AtomGroup(i1, self), AtomGroup(i2, self), AtomGroup(i3, self)

    def select_compounds(self, selection: str, compound: str = "residues"):
        ag_list = []
        if compound == "group":
            ag = self.select_atoms(selection)
            if len(ag) != 0:
                ag_list.append(ag)
        elif compound in ["residues", "molecules", "fragments", "segments"]:
            for c in getattr(self, compound):
                ag = c.atoms.select_atoms(selection)
                if len(ag) != 0:
                    ag_list.append(ag)
        else:
            raise MDStuffError(f"invalid parameter argument: {compound=}")

        return ag_list
