import collections
from typing import Any, List

import MDAnalysis as mda

def setup_universe(pfs: str, dcds: Any[str, List]) -> mda.Universe:
    """
    Setup the mdanalysis universe.

    :param pfs: the psf file
    :param dcds: a sequence of dcd files
    :return: the mdanalysis universe
    """
    u = mda.Universe(psf)



def var2list(var: Any) -> List:
    """
    Convert a Python variable to a list containing that variable, unless it is already iterable (but not a string),
    in which case the returned list will contain the contents of the variable.

    :param var: any Python variable reference
    :return: a list
    """

    if isinstance(var, collections.Iterable) and not isinstance(var, str):
        return [iter(var)]
    else:
        return [var]
