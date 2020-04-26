from typing import Callable

import numpy as np

from mdstuff.core import CompoundArray, CompoundGroup, ParameterValueError


class Bonds:
    def __init__(
        self, compounds: CompoundArray, distance: bool = False, orientation: int = None
    ):
        self.compounds = compounds
        self.distance = distance

        if distance and orientation is not None:
            raise ParameterValueError(
                name="orientation",
                value=orientation,
                message="distance and orientation are mutually exclusive",
            )
        if orientation is not None and not 0 <= orientation <= 2:
            raise ParameterValueError(
                name="orientation", value=orientation, message="must be 0, 1, or 2",
            )
        self.orientation = orientation

    def __call__(self) -> np.ndarray:
        d = self.compounds.bonds()
        if self.distance:
            return np.linalg.norm(d, axis=1)
        elif self.orientation is not None:
            return d[:, self.orientation] / np.linalg.norm(d, axis=1)
        else:
            return d


class CenterOfMass:
    def __init__(self, compounds: CompoundGroup, axis: int = None):
        self.compounds = compounds
        if axis in [None, 0, 1, 2]:
            self.axis = axis
        else:
            raise ParameterValueError(
                name="axis", value=axis, message="should be None, 0, 1, or 2"
            )

    def __call__(self) -> np.ndarray:
        com = self.compounds.centers_of_mass()
        if self.axis is not None:
            return com[:, self.axis]
        else:
            return com


class Charges:
    def __init__(self, compounds: CompoundArray):
        self.compounds = compounds

    def __call__(self) -> np.ndarray:
        return self.compounds.charges()


class Dihedrals:
    def __init__(self, compounds: CompoundArray, radians: bool = False):
        self.compounds = compounds
        self.radians = radians

    def __call__(self) -> np.ndarray:
        d = self.compounds.dihedrals()
        if self.radians:
            return d
        else:
            return d * 180.0 / np.pi


class Dipoles:
    def __init__(
        self,
        compounds: CompoundArray,
        axis: int = None,
        magnitude: bool = False,
        orientation: bool = False,
    ):
        self.compounds = compounds

        if axis not in [None, 0, 1, 2]:
            raise ParameterValueError(
                name="axis", value=axis, message="should be 0, 1, or 2"
            )

        if magnitude and orientation:
            raise ParameterValueError(
                name="orientation",
                value=orientation,
                message="magnitude and orientation are mutually exclusive",
            )

        if orientation and axis is None:
            raise ParameterValueError(
                name="orientation",
                value=orientation,
                message="axis must be specified if orientation is requested",
            )

        self.axis = axis
        self.magnitude = magnitude
        self.orientation = orientation

    def __call__(self) -> np.ndarray:
        d = self.compounds.dipoles()
        if self.axis is None:
            if self.magnitude:
                return np.linalg.norm(d, axis=1)
            else:
                return d
        else:
            if self.orientation:
                return d[:, self.axis] / np.linalg.norm(d, axis=1)
            else:
                return d[:, self.axis]


class Masses:
    def __init__(self, compounds: CompoundArray):
        self.compounds = compounds

    def __call__(self) -> np.ndarray:
        return self.compounds.masses()


class Positions:
    def __init__(self, compounds: CompoundArray, axis: int = None):
        self.compounds = compounds
        if axis in [None, 0, 1, 2]:
            self.axis = axis
        else:
            raise ParameterValueError(
                name="axis", value=axis, message="should be None, 0, 1, or 2"
            )

    def __call__(self) -> np.ndarray:
        x = self.compounds.positions()
        if self.axis is not None:
            return x[:, :, self.axis]
        else:
            return x


class PrincipalAxes:
    def __init__(self, compounds: CompoundGroup, n: int = 1, orientation: int = None):
        self.compounds = compounds

        if n not in [1, 2, 3]:
            raise ParameterValueError(name="n", value=n, message="should be 1, 2, or 3")
        self.n = n

        if orientation is not None and not 0 <= orientation <= 2:
            raise ParameterValueError(
                name="orientation", value=orientation, message="should be 0, 1, or 2"
            )
        self.orientation = orientation

    def __call__(self) -> np.ndarray:
        axes = self.compounds.principal_axes()[:, 3 - self.n, :]
        if self.orientation is not None:
            return axes[:, self.orientation] / np.linalg.norm(axes, axis=1)
        else:
            return axes


class PrincipalMoments:
    def __init__(self, compounds: CompoundGroup, n: int = 1):
        self.compounds = compounds
        if n not in [1, 2, 3]:
            raise ParameterValueError(name="n", value=n, message="should be 1, 2, or 3")
        self.n = n

    def __call__(self) -> np.ndarray:
        moments = self.compounds.principal_moments()[:, 3 - self.n]
        return moments


class TotalMass:
    def __init__(self, compounds: CompoundGroup):
        self.compounds = compounds

    def __call__(self) -> np.ndarray:
        return self.compounds.total_mass()


class UserFunction:
    def __init__(self, fn: Callable, *args, **kwargs):
        if not callable(fn):
            raise ParameterValueError(name="fn", value=fn, message="should be callable")
        self.fn = fn
        self.args = args
        self.kwargs = kwargs

    def __call__(self):
        return self.fn(*self.args, **self.kwargs)
