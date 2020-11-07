from __future__ import annotations

from typing import List, Sequence, Union

import MDAnalysis
import numpy as np
import tqdm

from .analyses import Analysis
from . import ParallelAnalysis, SerialAnalysis
from .errors import UniverseError, ParameterValueError, InputError
from ..transformations.transformations import Transformation


class ContinuousDCDReader(MDAnalysis.coordinates.chain.ChainReader):
    """
    A continuous DCD reader. The implementation is based on the MDAnalysis ChainReader.

    Differences are:
        - Only supports DCD files.
        - Trajectory lengths must be given as a sequence (None means using all frames).
        - Enforces a uniform time step.
        - No skip parameter.
    """

    format = "CONTINUOUSDCD"

    def __init__(self, filenames: Sequence[str], lengths: Sequence[int], **kwargs):
        """
        Initialize the reader.

        Args:
            filenames (Sequence[str]): The file names.
            lengths (Sequence[int]): The trajectory lengths.

        Raises:
            InputError: A file cannot be read.
        """
        # We're overwriting what happens ChainReader.__init__(), so we need to explicitly call the
        # grand-parent initializer.
        # pylint: disable=bad-super-call
        super(MDAnalysis.coordinates.chain.ChainReader, self).__init__()

        # References to all readers and filenames.
        self.readers = []
        for f in filenames:
            try:
                self.readers.append(
                    MDAnalysis.coordinates.DCD.DCDReader(filename=f, **kwargs)
                )
            except OSError:
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
        Return the cumulative time of all frames.

        Returns:
            float: The time.
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
        """
        Initialize a universe.

        Args:
            topology (str): The topology file.
            trajectories (Sequence[str]): The trajectory files.
            lengths (Sequence[int], optional): The trajectory lengths. Defaults to None.

        Raises:
            UniverseError: If there is more than one Universe.
            ParameterValueError: If a parameter has an invalid value.
            InputError: If there is a mismatch between the topology and the trajectories.
        """

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
        Register transformations.
        """
        self.transformations += transformations

    def apply_transformations(self):
        """
        Apply all registered transformations.
        """
        for t in self.transformations:
            t.apply(self.universe.atoms)

    def add_analyses(self, *analyses: Analysis):
        """
        Register analyses.
        """
        for analysis in analyses:
            if isinstance(analysis, ParallelAnalysis):
                self.parallel_analyses.append(analysis)
            elif isinstance(analysis, SerialAnalysis):
                self.serial_analyses.append(analysis)

    def run_analyses(self, start: int = None, stop: int = None, step: int = None):
        """
        Run all registered analyses.

        Args:
            start (int, optional): The first time step. Defaults to None.
            stop (int, optional): The last time step. Defaults to None.
            step (int, optional): The time step increment. Defaults to None.
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
        if len(self.serial_analyses) != 0:
            for analysis in self.serial_analyses:
                analysis.run(universe=self)
            self.serial_analyses = []

    def select_compounds(
        self, *selection: str, group_by: str = "residues", **kwargs
    ) -> CompoundGroup:
        """
        Select compounds (groups of atoms).

        Args:
            *selections (str): The selection strings. Use multiple strings if the order is important.
            group_by (str, optional): The compound type by which atoms will be grouped. Defaults to "residues".
            **kwargs: Keyword arguments passed to MDAnalysis.Universe.select_atoms().

        Raises:
            ParameterValueError: If a parameters has an invalid value.

        Returns:
            CompundGroup: Returns a CompoundGroup instance or an instance of its subtype, CompoundArray.
        """
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
        Create a compound group.

        Args:
            universe (MDAnalysis.Universe): The universe.
            ag_list (List[MDAnalysis.core.groups.AtomGroup]): The atom groups containing each compound.
        """
        self.universe = universe
        self.compounds = ag_list

    def __repr__(self) -> str:
        return f"<{self.__class__.__name__} with {len(self.compounds)} compounds>"

    def __add__(self, other):
        return self.__class__(
            universe=self.universe, ag_list=self.compounds + other.compounds
        )

    def centers_of_geometry(self) -> np.ndarray:
        """
        Compute the centers of geometry.

        Returns:
            np.ndarray: The centers of geometry.
        """
        return np.array([compound.center_of_geometry() for compound in self.compounds])

    def centers_of_mass(self) -> np.ndarray:
        """
        Compute the centers of masses.

        Returns:
            np.ndarray: The centers of masses.
        """
        return np.array([compound.center_of_mass() for compound in self.compounds])

    def dipoles(self) -> np.ndarray:
        """
        Compute the dipoles.

        Returns:
            np.ndarray: The dipoles.
        """
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
        """
        Compute the principal axes.

        Returns:
            np.ndarray: The principal axes.
        """
        return np.array(
            [
                (np.linalg.eigh(compound.moment_of_inertia())[1]).T
                for compound in self.compounds
            ]
        )

    def principal_moments(self) -> np.ndarray:
        """
        Compute the principal moments.

        Returns:
            np.ndarray: The principal moments.
        """
        return np.array(
            [
                np.linalg.eigvalsh(compound.moment_of_inertia())
                for compound in self.compounds
            ]
        )

    def total_mass(self) -> np.ndarray:
        """
        Compute the total mass.

        Returns:
            np.ndarray: The total mass.
        """
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
        Initialize a compound array.

        Args:
            universe (MDAnalysis.Universe): The universe.
            ag_list (List[MDAnalysis.core.groups.AtomGroup]): A list of atom groups.
        """

        super().__init__(universe=universe, ag_list=ag_list)

        # An array storing the atom indices.
        self.indices = np.array([ag.ix for ag in self.compounds])
        # The shape of the CompoundGroup (nr. of compounds, nr. of atoms per compound)
        self.shape = self.indices.shape

    def bonds(self) -> np.ndarray:
        """
        Compute bond vectors.

        Raises:
            InputError: If the number of atoms per compound differs from 2.

        Returns:
            np.ndarray: The bond vectors.
        """
        if self.shape[1] != 2:
            raise InputError(
                "all compounds must consist of two atoms to compute a bond"
            )

        x1, x2 = self.universe.atoms.positions[self.indices.T]
        return x2 - x1

    def centers_of_geometry(self) -> np.ndarray:
        """
        Compute the centers of geometry.

        Returns:
            np.ndarray: The centers of geometry.
        """
        x = self.universe.atoms.positions[self.indices]
        return np.mean(x, axis=1)

    def centers_of_mass(self) -> np.ndarray:
        """
        Compute the centers of geometry.

        Returns:
            np.ndarray: The centers of geometry.
        """
        x = self.universe.atoms.positions[self.indices]
        m = self.universe.atoms.masses[self.indices]
        return np.sum(x * m[:, :, None], axis=1) / np.sum(m, axis=1)[:, None]

    def charges(self) -> np.ndarray:
        """
        Return the atomic charges.

        Returns:
            np.ndarray: The charges.
        """
        return self.universe.atoms.charges[self.indices]

    def dihedrals(self) -> np.ndarray:
        """
        Compute dihedral angles.

        Raises:
            InputError: If the number of atoms per compound differs from 4.

        Returns:
            np.ndarray: The dihedral angles.
        """
        if self.shape[1] != 4:
            raise InputError(
                "all compounds must consist of four atoms to compute a dihedral"
            )

        x1, x2, x3, x4 = self.universe.atoms.positions[self.indices.T]
        b1 = x2 - x1
        b2 = x3 - x2
        b3 = x4 - x3
        n1 = np.cross(b1, b2)
        n2 = np.cross(b2, b3)
        n1 /= np.linalg.norm(n1, axis=1)[:, None]
        n2 /= np.linalg.norm(n2, axis=1)[:, None]
        m1 = np.cross(n1, b2)
        x = np.sum(n1 * n2, axis=1)
        y = np.sum(m1 * n2, axis=1)
        return np.arctan2(y, x) * 180.0 / np.pi

    def dipoles(self) -> np.ndarray:
        """
        Compute the dipoles.

        Returns:
            np.ndarray: The dipoles.
        """
        x = self.universe.atoms.positions[self.indices]
        q = self.universe.atoms.charges[self.indices]
        mu = np.sum(q[:, :, None] * x, axis=1)
        return mu

    def masses(self) -> np.ndarray:
        """
        Return the atomic masses.

        Returns:
            np.ndarray: The masses.
        """
        return self.universe.atoms.masses[self.indices]

    def positions(self) -> np.ndarray:
        """
        Return the atomic positions.

        Returns:
            np.ndarray: The positions.
        """
        return self.universe.atoms.positions[self.indices]
