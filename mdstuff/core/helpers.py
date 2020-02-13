from collections import Sequence
from typing import List


def to_list(obj: object) -> List:
    """
    Convert an object to a list.
    If the object is already a list, return the object.
    If the object is any non-string sequence, return a list containing all sequence elements.
    Otherwise, return a list containing the object.
    """
    if obj is None:
        return None
    elif isinstance(obj, list):
        return obj
    elif isinstance(obj, Sequence) and not isinstance(obj, str):
        return list(obj)
    else:
        return [obj]
