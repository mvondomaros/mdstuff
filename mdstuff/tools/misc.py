import collections
from typing import Any, List


def aslist(obj: Any) -> List:
    """
    Convert a Python object to a list containing that object, unless it is an iterable (but not a string),
    in which case the returned list will contain all elements of the iterable.

    :param obj: any Python object
    :return: a list
    """

    if isinstance(obj, collections.Iterable) and not isinstance(obj, str):
        return [iter(obj)]
    else:
        return [obj]
