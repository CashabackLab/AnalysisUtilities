from numba import njit
import numpy as np

@nb.njit
def np_apply_along_axis(func1d, arr, axis):
    assert axis in [0, 1]

    if arr.ndim == 1:
        result = func1d(arr)
    else:            
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
    return np_apply_along_axis(np.nanstd, array, axis)

@njit
def nb_mean(array, axis):
    return np_apply_along_axis(np.mean, array, axis)

@njit
def nb_nanmean(array, axis):
    return np_apply_along_axis(np.nanmean, array, axis)

@nb.njit
def nb_nanmedian(array, axis):
    return np_apply_along_axis(np.nanmedian, array, axis)
