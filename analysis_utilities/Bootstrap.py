from tqdm.notebook import tqdm
import numpy as np
from numpy.random import default_rng
import scipy.stats as stats
from numba import jit, njit
import numba as nb
from numba.core.errors import NumbaDeprecationWarning, NumbaPendingDeprecationWarning
import warnings

warnings.simplefilter('ignore', category=NumbaDeprecationWarning)
warnings.simplefilter('ignore', category=NumbaPendingDeprecationWarning)

def _Two_Tailed_Bootstrap(data1, data2, M = 1e4, paired = False, verbose = False):
    M = int(M)
    rng = default_rng()
    #data1, data2 = np.array(data1), np.array(data2)
    assert data1.shape == data2.shape
    results = np.empty(M) * np.nan
    data_len = data1.shape[0]
    pooled_data = np.empty(data1.shape[0] + data2.shape[0]) * np.nan
    pooled_data[:data1.shape[0]] = data1
    pooled_data[data1.shape[0]:] = data2
    
    if paired:
        paired_diff = data1 - data2
        for i in tqdm(range(M)):
            results[i] = np.nanmean(rng.choice(paired_diff, size = data_len, replace = True))
    else:
        for i in tqdm(range(M)):
            tmp = rng.choice(pooled_data, size = data_len*2, replace = False)
            data1_resample = tmp[:data_len]
            data2_resample = tmp[data_len:]
            mean_diff = np.nanmean(data1_resample) - np.nanmean(data2_resample)
            results[i] = mean_diff
    
    if paired: 
        paired_diff = data1 - data2
        original_mean_diff = np.nanmean(paired_diff)
        centered = results - original_mean_diff
    else: 
        original_mean_diff = np.nanmean(data1) - np.nanmean(data2)
        centered = results
        
    p_val = np.sum(centered > abs(original_mean_diff)) + np.sum(centered < -abs(original_mean_diff))
    p_val /= M
    
    if verbose:
        return centered, p_val
    else:
        return p_val
    
@njit(parallel = True)
def _NB_Two_Tailed_Bootstrap(data1, data2, M = 1e4, paired = False, verbose = False):
    M = int(M)
    rng = np.random
    #data1, data2 = np.array(data1), np.array(data2)
    # assert data1.shape == data2.shape
    results = np.empty(M) * np.nan
    data_len = data1.shape[0]
    pooled_data = np.empty(data1.shape[0] + data2.shape[0]) * np.nan
    pooled_data[:data1.shape[0]] = data1
    pooled_data[data1.shape[0]:] = data2
    original_mean_diff = -1

    if paired:
        paired_diff = data1 - data2
        for i in nb.prange(M):
            results[i] = np.nanmean(rng.choice(paired_diff, size = data_len, replace = True))
    else:        
        for i in range(M): #cannot be parallelized
            tmp = rng.choice(pooled_data, size = data_len*2, replace = False)
            data1_resample = tmp[:data_len]
            data2_resample = tmp[data_len:]
            mean_diff = np.nanmean(data1_resample) - np.nanmean(data2_resample)
            results[i] = mean_diff
    
    if paired: 
        paired_diff = data1 - data2
        original_mean_diff = np.nanmean(paired_diff)
        centered = results - original_mean_diff
    else: 
        original_mean_diff = np.nanmean(data1) - np.nanmean(data2)
        centered = results
    
    p_val = np.sum(centered > abs(original_mean_diff)) + np.sum(centered < -abs(original_mean_diff))
    p_val /= M
    
    return p_val

def _One_Tailed_Bootstrap(data1, data2, M = 1e4, paired = False, direction = "lesser", verbose = False):
    M = int(M)
    #data1, data2 = np.array(data1), np.array(data2)
    test_results = np.zeros(M) * np.nan
    rng = np.random.default_rng()
    
    if paired:
        paired_diff = data1 - data2

        for i in tqdm(range(M)):
            diff_resampled = rng.choice(paired_diff, size = len(paired_diff), replace = True)

            test_results[i] = np.nanmean(diff_resampled)        
    else:    
        for i in tqdm(range(M)):
            data1_resampled = rng.choice(data1, size = len(data1), replace = True)
            data2_resampled = rng.choice(data2, size = len(data2), replace = True)

            diff = data1_resampled - data2_resampled
            test_results[i] = np.nanmean(diff)
            
    if direction == "lesser": p_val = np.sum(test_results < 0) / M
    else : p_val = np.sum(test_results > 0) / M
    
    if verbose:
        return test_results, p_val
    else:
        return p_val
    
@njit(parallel = True)
def _NB_One_Tailed_Bootstrap(data1, data2, M = 1e4, paired = False, direction = "lesser", verbose = False):
    M = int(M)
    #data1, data2 = np.array(data1), np.array(data2)
    test_results = np.zeros(M) * np.nan
    rng = np.random
    
    if paired:
        paired_diff = data1 - data2
        for i in nb.prange(M):
            diff_resampled = rng.choice(paired_diff, size = len(paired_diff), replace = True)

            test_results[i] = np.nanmean(diff_resampled)        
    else:    
        for i in nb.prange(M):
            data1_resampled = rng.choice(data1, size = len(data1), replace = True)
            data2_resampled = rng.choice(data2, size = len(data2), replace = True)

            diff = data1_resampled - data2_resampled 
            test_results[i] = np.nanmean(diff)
            
    if direction == "lesser": p_val = np.sum(test_results < 0) / M
    else : p_val = np.sum(test_results > 0) / M

    return p_val
    

def Bootstrap(data1, data2, M = 1e4, paired = False, direction = None, verbose = False):
    """ Bootstrap difference in means between two groups.
    M = float64 # Number of iterations
    paired = {True, False}
    direction = {"greater", "lesser", None} 
    if verbose = True, returns distribution and p_value, shows progress bar
    if not verbose, only returns p-value, suppresses progress bar and uses numba"""
    data1, data2 = np.array(data1), np.array(data2)
    if not verbose:
        if direction == None or direction == "two-tailed":
            return _NB_Two_Tailed_Bootstrap(data1, data2, M = M, paired = paired, verbose = verbose)
        else:
            return _NB_One_Tailed_Bootstrap(data1, data2, M = M, paired = paired, direction = direction,  verbose = verbose)
    else:
        if direction == None or direction == "two-tailed":
            return _Two_Tailed_Bootstrap(data1, data2, M = M, paired = paired, verbose = verbose)
        else:
            return _One_Tailed_Bootstrap(data1, data2, M = M, paired = paired, direction = direction,  verbose = verbose)
