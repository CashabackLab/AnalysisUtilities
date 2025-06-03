from numba import njit
import numpy as np

@njit
def np_apply_along_axis(func1d, axis, arr):
    assert arr.ndim == 2
    assert axis in [0, 1]
    if axis == 0:
        result = np.empty(arr.shape[1])
        for i in range(len(result)):
            result[i] = func1d(arr[:, i])
    else:
        result = np.empty(arr.shape[0])
        for i in range(len(result)):
            result[i] = func1d(arr[i, :])
    return result

@njit
def nb_nanstd(array, axis):
    return np_apply_along_axis(np.nanstd, axis, array)

@njit
def nb_mean(array, axis):
    return np_apply_along_axis(np.mean, axis, array)

@njit
def nb_nanmean(array, axis):
    return np_apply_along_axis(np.nanmean, axis, array)

@njit
def nb_nanmedian(array, axis):
    return np_apply_along_axis(np.nanmedian, axis, array)
