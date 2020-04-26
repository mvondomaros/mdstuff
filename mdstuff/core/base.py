from typing import List, Sequence

import MDAnalysis
import numpy as np
import tqdm

from .analyses import Analysis
from . import ParallelAnalysis
from .errors import UniverseError, ParameterValueError, InputError
from ..transformations.transformations import Transformation


class ContinuousDCDReader(MDAnalysis.coordinates.chain.ChainReader):
    """
    A continuous DCD reader. The implementation is based on the MDAnalysis ChainReader.
    Differences are:
        - only supports DCD files
        - trajectory lengths must be given as a sequence (None means using all frames)
        - enforces a uniform time step
        - no skip parameter
    """

    format = "CONTINUOUSDCD"

    def __init__(self, filenames: Sequence[str], lengths: Sequence[int], **kwargs):
        """
        :param filenames: a sequence of file names
        :param lengths: a sequence of trajectory lengths
        """
        # We're overwriting what happens ChainReader.__init__(), so we need to explicitly call the
        # grand-parent initializer.
        super(MDAnalysis.coordinates.chain.ChainReader, self).__init__()

        # References to all readers and filenames.
        self.readers = []
        for f in filenames:
            try:
                self.readers.append(
                    MDAnalysis.coordinates.DCD.DCDReader(filename=f, **kwargs)
                )
            except OSError as e:
                raise InputError(f"could not read DCD file: {f}")
        self.filenames = np.array(filenames)

        # The index of the active reader.
        self.__active_reader_index = 0

        # We enforce an equal time step and equal number of atoms.
        self.dts = self._get("dt")  # Needed as an array by some inherited methods.
        self._get_same("dt")
        self.n_atoms = self._get_same("n_atoms")

        # Get the total number of frames.
        n_total_frames = self._get("n_frames")
        # Get the desired number of frames specified through lengths. Use all frames, if None is given.
        n_desired_frames = [
            length if length is not None else n
            for length, n in zip(lengths, n_total_frames)
        ]

        # The total number of desired frames.
        self.n_frames = np.sum(n_desired_frames)
        # The virtual indices corresponding to the start of each trajectory.
        self._start_frames = np.cumsum([0] + n_desired_frames)
        # The cumulative time passed after each individual trajectory.
        self.total_times = self.dt * np.array(n_desired_frames)
        # Technical stuff copied shamelessly from the base class.
        self.__chained_trajectories_iter = None
        self.ts = None
        self.rewind()

    @property
    def time(self) -> float:
        """
        Cumulative time of all frames in MDAnalysis time units (typically ps).

        :return: the time
        """
        traj_index, sub_frame = self._get_local_frame(self.frame)
        # Added +1 to sub_frame, since DCD files do not contain the initial coordinates.
        return (
            self.total_times[:traj_index].sum() + (sub_frame + 1) * self.dts[traj_index]
        )

    # Needed because of ABC requirements, but not implemented.
    def Writer(self, *args, **kwargs):
        raise NotImplementedError

    # Needed because of ABC requirements, but not implemented.
    @classmethod
    def parse_n_atoms(cls, *args, **kwargs):
        raise NotImplementedError

    @property
    def dt(self) -> float:
        """
        Returns the time step.

        :return: the time step
        """
        return self.dts[0]


class Universe(MDAnalysis.Universe):
    """
    Base universe class.
    """

    # Singleton instance reference.
    instance = None

    def __init__(
        self,
        topology: str,
        trajectories: Sequence[str],
        lengths: Sequence[int] = None,
        **kwargs,
    ):
        # Check if there's already another universe.
        if Universe.instance is not None:
            raise UniverseError("there can only be one universe")
        else:
            Universe.instance = self

        # Check if a normal MDAnalysis will do.
        if lengths is None:
            super().__init__(topology, *trajectories, **kwargs)
        else:
            super().__init__(topology, **kwargs)

            if len(trajectories) != len(lengths):
                raise ParameterValueError(
                    name="lengths",
                    value=lengths,
                    message="must have the same number of items as trajectories",
                )
            reader = ContinuousDCDReader(trajectories, lengths)

            # Compare the number of atoms.
            self.n_atoms = len(self.atoms)
            if self.n_atoms != reader.n_atoms:
                raise InputError(
                    f"the number of atoms in the PSF and DCD files do not match ({self.n_atoms} != {reader.n_atoms})"
                )
            self.trajectory = reader

        # The analysis and transformation queues.
        self.parallel_analyses = []
        self.serial_analyses = []
        self.transformations = []

    # Needed because of ABC requirements, but not implemented.
    def __getstate__(self, *args, **kwargs):
        raise NotImplementedError

    # Needed because of ABC requirements, but not implemented.
    def __setstate__(self, *args, **kwargs):
        raise NotImplementedError

    def add_transformations(self, *transformations: Transformation):
        """
        Add transformations.

        :param transformations: the transformations
        """
        self.transformations += transformations

    def apply_transformations(self):
        for t in self.transformations:
            t.apply(self.universe.atoms)

    def add_analyses(self, *analyses: Analysis):
        """
        Add analyses.

        :param analyses: the analyses
        """
        for analysis in analyses:
            if isinstance(analysis, ParallelAnalysis):
                self.parallel_analyses.append(analysis)

    def run_analyses(self, start: int = None, stop: int = None, step: int = None):
        """
        Run all analyses.

        :param start: optional, the first time step
        :param stop: optional, the last time step
        :param step: optional, the time step increment
        """
        universe = Universe.instance

        if len(self.parallel_analyses) != 0:
            for _ in tqdm.tqdm(
                universe.trajectory[start:stop:step], desc="main trajectory loop"
            ):
                self.apply_transformations()
                for analysis in self.parallel_analyses:
                    analysis.update()
            self.parallel_analyses = []

    def select_compounds(self, *selection: str, group_by: str = "residues", **kwargs):
        supported_groups = ["residues"]
        if group_by not in supported_groups:
            raise ParameterValueError(
                name="group_by",
                value=group_by,
                message=f"supported values are {', '.join(supported_groups)}",
            )

        updating = kwargs.setdefault("updating", False)

        ag_list = []
        for compound in getattr(self, group_by):
            ag = compound.atoms.select_atoms(*selection, **kwargs)
            if len(ag) != 0:
                ag_list.append(ag)

        nr_atoms = [len(ag) for ag in ag_list]
        if not updating and len(np.unique(nr_atoms)) == 1:
            return CompoundArray(universe=self, ag_list=ag_list)
        else:
            return CompoundGroup(universe=self, ag_list=ag_list)


class CompoundGroup:
    """
    A group of compounds (e.g., atoms grouped by residues).
    """

    def __init__(
        self,
        universe: MDAnalysis.Universe,
        ag_list: List[MDAnalysis.core.groups.AtomGroup],
    ):
        """
        :param universe: a Universe
        :param ag_list: a list of MDAnalysis AtomGroups
        """
        self.universe = universe
        self.compounds = ag_list

    def __repr__(self) -> str:
        return f"<{self.__class__.__name__} with {len(self.compounds)} compounds>"

    def __add__(self, other):
        return self.__class__(
            universe=self.universe, ag_list=self.compounds + other.compounds
        )

    def centers_of_mass(self) -> np.ndarray:
        return np.array([compound.center_of_mass() for compound in self.compounds])

    def dipoles(self) -> np.ndarray:
        x = np.array(
            [
                compound.positions - compound.center_of_mass()
                for compound in self.compounds
            ]
        )
        q = np.array([compound.charges for compound in self.compounds])
        mu = np.sum(q[:, :, None] * x, axis=1)
        return mu

    def principal_axes(self) -> np.ndarray:
        return np.array(
            [
                (np.linalg.eigh(compound.moment_of_inertia())[1]).T
                for compound in self.compounds
            ]
        )

    def principal_moments(self) -> np.ndarray:
        return np.array(
            [
                np.linalg.eigvalsh(compound.moment_of_inertia())
                for compound in self.compounds
            ]
        )

    def total_mass(self) -> np.ndarray:
        return np.array([compound.total_mass() for compound in self.compounds])


class CompoundArray(CompoundGroup):
    """
    A compound group where each compound has the same number of atoms.

    This should allow for the implementation of faster, specialized methods.
    """

    def __init__(
        self,
        universe: MDAnalysis.Universe,
        ag_list: List[MDAnalysis.core.groups.AtomGroup],
    ):
        """
        :param universe: a Universe
        :param ag_list: a list of MDAnalysis AtomGroups
        """

        super().__init__(universe=universe, ag_list=ag_list)

        # An array storing the atom indices.
        self.indices = np.array([ag.ix for ag in self.compounds])
        # The shape of the CompoundGroup (nr. of compounds, nr. of atoms per compound)
        self.shape = self.indices.shape

    def bonds(self) -> np.ndarray:
        if self.shape[1] != 2:
            raise InputError(
                "all compounds must consist of two atoms to compute a bond"
            )

        x1, x2 = self.universe.atoms.positions[self.indices.T]
        return x2 - x1

    def charges(self) -> np.ndarray:
        return self.universe.atoms.charges[self.indices]

    def dihedrals(self) -> np.ndarray:
        if self.shape[1] != 4:
            raise InputError(
                "all compounds must consist of four atoms to compute a dihedral"
            )

        x1, x2, x3, x4 = self.universe.atoms.positions[self.indices.T]
        a = x1 - x2
        b = x4 - x3
        c = x3 - x2
        axc = np.cross(a, c)
        bxc = np.cross(b, c)
        axc_norm = np.linalg.norm(axc, axis=1)
        bxc_norm = np.linalg.norm(bxc, axis=1)
        cos = np.sum(axc * bxc, axis=1) / (axc_norm * bxc_norm)
        # Sometimes there are underflows, because of numerical imprecision.
        np.maximum(cos, -1.0, out=cos)
        # And sometimes there are overflows.
        np.minimum(cos, 1.0, out=cos)
        return np.arccos(cos)

    def masses(self) -> np.ndarray:
        return self.universe.atoms.masses[self.indices]

    def positions(self) -> np.ndarray:
        return self.universe.atoms.positions[self.indices]
