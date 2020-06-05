from typing import Any


class MDStuffError(Exception):
    """
    Base error class in mdstuff.
    """

    pass


class UniverseError(MDStuffError):
    """
    Raised when no or multiple universes were created.
    """

    pass


class InputError(MDStuffError):
    """
    Raised when something is wrong with the topology/trajectory input.
    """

    pass


class ParameterValueError(MDStuffError):
    """
    Raised when a parameter takes on an invalid value.
    """

    def __init__(self, name: str, value: Any, message: str = None):
        """
        Initialize a parameter value error.

        Args:
            name (str): The name of the parameter.
            value (Any): Its value.
            message (str, optional): A descriptive message. Defaults to None.
        """        
        super().__init__()
        self.name = name
        self.value = repr(value) if isinstance(value, str) else value
        self.message = message

    def __str__(self):
        s = f"invalid parameter value ({self.name}={self.value})"
        if self.message is not None:
            s += f"; {self.message}"
        return s


class AnalysisError(MDStuffError):
    """
    Raised when something goes wrong during an analysis.
    """

    pass
