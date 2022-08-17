from numba import njit
import numpy as np

@njit
def serial_corr(arr, lag=1):
    n = len(arr)
    y1 = arr[lag:]
    y2 = arr[:n-lag]
    corr = np.corrcoef(y1, y2)[0, 1]
    return corr
