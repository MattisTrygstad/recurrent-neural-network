import numpy as np


def shuffle_arrays(a: np.ndarray, b: np.ndarray) -> tuple:
    assert len(a) == len(b)
    p = np.random.permutation(len(a))
    return a[p], b[p]
