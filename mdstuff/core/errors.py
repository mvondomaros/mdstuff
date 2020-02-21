import collections
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

    def __init__(self, name: str, value: Any, allowed_values: Any, *args, **kwargs):
        """
        :param name: the name of the parameter
        :param value: the value of the parameter
        :param allowed_values: the allowed values
        """
        super().__init__(*args, **kwargs)
        self.name = name
        self.value = repr(value) if isinstance(value, str) else value
        self.allowed_values = allowed_values

    def __str__(self):
        if isinstance(self.allowed_values, str):
            what = self.allowed_values
        elif isinstance(self.allowed_values, collections.Iterable):
            what = f"one of {self.allowed_values}"
        else:
            what = f"{self.allowed_values}"
        message = f"invalid parameter value ({self.name}={self.value}); expected {what}"
        return message
