from __future__ import annotations

import abc
import itertools
from typing import Iterable, Union, Any, Tuple

import MDAnalysis
import numpy as np
import tqdm

from .errors import MDStuffError
from .helpers import to_list


class Analysis:
    """Base analysis class."""

    def __init__(self):
        """Initializations that do not depend on the universe being set"""
        self.universe = None

    def _late_init(self):
        """Initializations that depend on the universe being set."""
        pass

    def set_universe(self, universe: Universe):
        """
        Set the universe and perform late initializations.

        :param universe: the universe
        """
        self.universe = universe
        self._late_init()

    @abc.abstractmethod
    def update(self, *args, **kwargs):
        """Update the analysis."""
        pass

    @abc.abstractmethod
    def get(self, *args, **kwargs) -> Any:
        """
        Return the results.

        :return: the results
        """
        pass


class Universe(MDAnalysis.Universe):
    """The MDStuff universe. It's basically an MDAnalysis universe with an analysis handling mechanism."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.analyses = []

    def __getstate__(self, *args, **kwargs):
        raise NotImplementedError

    def __setstate__(self, *args, **kwargs):
        raise NotImplementedError

    def add_analyses(self, analyses: Union[Analysis, Iterable[Analysis]]):
        """
        Add analyses.

        :param analyses: the analyses or an iterable over analyses
        """
        for analysis in to_list(analyses):
            analysis.set_universe(self)
            self.analyses.append(analysis)

    def run_analyses(self, start: int = 0, stop: int = -1, step: int = 1):
        """
        Run all analyses.

        :param start: optional, the first time step
        :param stop: optional, the last time step
        :param step: optional, the time step increment
        """
        for _ in tqdm.tqdm(self.trajectory[start:stop:step]):
            for analysis in self.analyses:
                analysis.update()

    def select_atom_pairs(
        self,
        selection1: str,
        selection2: str,
        mode: str = "zip",
        compound: str = None,
        updating=False,
    ) -> Tuple[MDAnalysis.core.groups.AtomGroup, MDAnalysis.core.groups.AtomGroup]:
        if compound is not None and compound not in [
            "residues",
            "molecules",
            "fragments",
            "segments",
        ]:
            raise MDStuffError(f"invalid parameter value: {compound=}")
        if mode not in ["zip", "product", "within", "between"]:
            raise MDStuffError(f"invalid parameter value: {mode=}")
        if mode in ["within", "between"] and compound is None:
            raise MDStuffError(f"must specify the compound parameter, if '{mode=}'")
        if updating is True:
            raise MDStuffError(
                f"atom pair selection is static; updates are not permitted"
            )

        index_set = set()
        index_list = list()

        if mode == "zip" or mode == "product":
            ag1 = self.select_atoms(selection1)
            ag2 = self.select_atoms(selection2)
            n1 = len(ag1)
            n2 = len(ag2)
            if mode == "zip":
                if n1 != n2:
                    raise MDStuffError(
                        f"the number of atoms in selection1 ({n1}) does not match the number of atoms in selection2 ({n2})"
                    )
                elif n1 == 0:
                    raise MDStuffError(f"selection1 and selection2 are empty")
            elif mode == "product":
                if n1 == 0:
                    raise MDStuffError(f"selection1 is empty")
                if n2 == 0:
                    raise MDStuffError(f"selection2 is empty")
                for i1, i2 in itertools.product(ag1.ix, ag2.ix):
                    if (i1, i2) not in index_set:
                        index_set.add((i1, i2))
                        index_list.append((i1, i2))
            else:
                raise NotImplementedError
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
        return (
            MDAnalysis.core.groups.AtomGroup(i1, self),
            MDAnalysis.core.groups.AtomGroup(i2, self),
        )
