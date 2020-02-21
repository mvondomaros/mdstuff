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


def same_universe(*ag_list) -> bool:
    n = len(ag_list)
    if n == 0 or n == 1:
        return True
    else:
        u = ag_list[0].universe
        for ag in ag_list[1:]:
            if ag.universe != u:
                return False
        return True
