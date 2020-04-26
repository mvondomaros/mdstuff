import abc

import MDAnalysis


class Transformation(abc.ABC):
    @abc.abstractmethod
    def apply(self, ag: MDAnalysis.core.groups.AtomGroup):
        pass


class Center(Transformation):
    def apply(self, ag: MDAnalysis.core.groups.AtomGroup):
        ag.positions -= ag.center_of_mass()
