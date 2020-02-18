import numpy as np
import abc

from .helpers import apply_mic
from ..core.errors import MDStuffError
from ..core.universe import Universe


class StructureFunction(abc.ABC):
    def __init__(self, universe: Universe):
        self.universe = universe

    @abc.abstractmethod
    def __call__(self, *args, **kwargs) -> np.ndarray:
        pass

    @property
    @abc.abstractmethod
    def shape(self) -> np.ndarray:
        pass


class Magnitude(StructureFunction):
    def __init__(self, function: StructureFunction):
        shape = function.shape
        if len(shape) != 2 or shape[1] != 3:
            raise MDStuffError(
                f"expected a function that returns vectors;"
                f" got a function that returns an array of shape {shape}"
            )

        super().__init__(universe=function.universe)

        self.function = function
        self._shape = np.array([shape[0]])

    def __call__(self) -> np.ndarray:
        return np.linalg.norm(self.function(), axis=1)

    @property
    def shape(self) -> np.ndarray:
        return self._shape


class Distance(StructureFunction):
    def __init__(self, ag1, ag2, use_mic: bool = True):
        n1 = len(ag1)
        n2 = len(ag2)
        if n1 != n2:
            raise MDStuffError(
                f"the number of atoms in ag1 ({n1}) does not match the number of atoms in ag2 ({n2})"
            )
        elif n1 == 0:
            raise MDStuffError(f"ag1 and ag2 are empty")

        super().__init__(ag1.universe)

        self.ag1 = ag1
        self.ag2 = ag2
        self.use_mic = use_mic

        self._shape = np.array([n1, 3])

    def __call__(self, *args, **kwargs):
        d = self.ag2.positions - self.ag1.positions
        if self.use_mic:
            apply_mic(d, self.universe.dimensions[:3])
        return d

    @property
    def shape(self) -> np.ndarray:
        return self._shape


# class ScalarF(abc.ABC):
#     def __call__(self, *args, **kwargs) -> np.ndarray:
#         pass
#
#
# class VectorF(abc.ABC):
#     def __call__(self, *args, **kwargs) -> np.ndarray:
#         pass
#
#
# class StructureF:
#     def __init__(self):
#         self.universe = None
#
#     def set_universe(self, universe: Universe):
#         self.universe = universe
#         self._late_init()
#
#     def _late_init(self):
#         pass
#
#     def __call__(self, *args, **kwargs) -> np.ndarray:
#         pass
#
#
# class ScalarF(StructureF):
#     pass
#
#
# class VectorF(StructureF):
#     pass
#
#
# class Magnitude(ScalarF):
#     def __init__(self, function: VectorF):
#
#         self.function = function
#
#     def __call__(self, *args, **kwargs):
#         vectors = self.function(*args, *kwargs)
#         return np.linalg.norm(vectors, axis=1)
#
#
# class Projection(ScalarF):
#     def __init__(self, function: VectorF, axis: int = 2):
#         self.function = function
#         if axis not in [0, 1, 2]:
#             raise MDStuffError(f"invalid parameter argument: {axis=}")
#         self.axis = axis
#
#     def __call__(self, *args, **kwargs):
#         vectors = self.function(*args, *kwargs)
#         return vectors[:, self.axis]
#
#
# class Orientation(Projection):
#     def __init__(self, function: VectorF, axis: int = 2):
#         super().__init__(function=function, axis=axis)
#
#     def __call__(self, *args, **kwargs):
#         vectors = self.function(*args, *kwargs)
#         return vectors[:, self.axis] / np.linalg.norm(vectors, axis=1)
#
#
# class Angle(ScalarF):
#     def __init__(self, function1: VectorF, function2: VectorF):
#         pass
