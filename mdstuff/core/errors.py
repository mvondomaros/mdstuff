from typing import Any


class MDStuffError(Exception):
    """Base error class in MDStuff."""

    pass


class DataError(MDStuffError):
    """
    Raised when something is wrong with the trajectory data.
    """

    def __init__(self, message: str):
        self.message = message


class UniverseError(MDStuffError):
    """
    MDStuff enforces a single universe. This exception is raised when a user attempts to violate this policy.
    """

    def __str__(self):
        return "there can only be one universe"


class ParameterValueError(MDStuffError):
    """
    Raised when a parameter takes on an invalid value.
    """

    def __init__(self, name: str, value: Any, allowed_values: Any):
        super().__init__()
        self.name = name
        self.value = value
        self.allowed_values = allowed_values

    def __str__(self):
        message = f"invalid parameter value: {self.name}={self.value}; expected "
