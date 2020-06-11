import numpy as np


def apply_mic(d: np.ndarray, box: np.ndarray):
    d -= np.round(d / box[:3]) * box[:3]

