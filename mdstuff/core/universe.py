from __future__ import annotations

import abc
from typing import Iterable, Union, Any

import MDAnalysis
import tqdm

from .helpers import to_list
from .types import Selection


class Analysis:
    """Base analysis class."""

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

    @abc.abstractmethod
    def update(self, *args, **kwargs) -> None:
        """Update the analysis."""
        pass

    @abc.abstractmethod
    def get(self, *args, **kwargs) -> Any:
        """Return the results."""
        pass


class Universe(MDAnalysis.Universe):
    """MDStuff Universe. It's basically an MDAnalysis Universe with an analysis handling mechanism."""

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.analyses = []

    def __getstate__(self):
        pass

    def __setstate__(self, state):
        pass

    def add_analyses(self, analyses: Union[Analysis, Iterable[Analysis]]) -> Universe:
        """
        :param analyses: an MDStuff analysis or an iterable thereof
        """
        for analysis in to_list(analyses):
            analysis.set_universe(self)
            self.analyses.append(analysis)
        return self

    def run_analyses(self, start: int = 0, stop: int = -1, step: int = 1) -> Universe:
        for _ in tqdm.tqdm(self.trajectory[start:stop:step]):
            for analysis in self.analyses:
                analysis.update()
        return self

    def nr_compounds(self, compound: str, selection: Selection = None) -> int:
        """
        Return the number of specified compounds.

        :param compound: a compound specifier
        :param selection: optional, a selection string
        :return: the number of compounds
        """
        if compound is None:
            ag = self.select_atoms(*selection)
            return len(ag)
        elif compound == "group":
            return 1
        else:
            if selection is None:
                return len(getattr(self, compound))
            else:
                n = 0
                selection = to_list(selection)
                for c in getattr(self, compound):
                    ag = c.atoms.select_atoms(*selection)
                    if len(ag) != 0:
                        n += 1
                return n
