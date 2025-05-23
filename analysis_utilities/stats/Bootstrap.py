import numpy as np
from numba import njit
import numba as nb
from numba.core.errors import NumbaDeprecationWarning, NumbaPendingDeprecationWarning
import warnings

warnings.simplefilter('ignore', category=NumbaDeprecationWarning)
warnings.simplefilter('ignore', category=NumbaPendingDeprecationWarning)

@njit(parallel = True)
def _NB_Bootstrap(data1, data2, M = 1e4, paired = False, alternative = "two-sided"):
    '''Performs a bootstrapped permutation test on our data. For a between test, will take two sets of 
    data and pool them. Two groups are then randomly sampled repetitively with replacement
    and a mean difference is calculated for each set of two groups sampled. This will then produce M bootstraps
    and a numerical null distribution. The p-value is the proportion of the data more extreme (either one or two sided)
    than our original data samples mean difference. 

    A paired test will resample from the pool of mean differences. In order to test a data set against zero, paired
    should be selected and the second data set will be zeros of the same length as the first data set. This will
    generate a null distribution for just the first data set.

    M = float64 # Number of iterations
    paired = {True, False}
    alternative = {"two-sided", "greater", "less"} #data1 relative to data2, i.e.: data1 "greater" than data2. 
    return_distribution {True, False} #returns the bootstrapped distribution
    '''
    rng = np.random

    results = np.empty(M) * np.nan
    original_mean_diff = np.nanmean(data1) - np.nanmean(data2)

    if paired:
        assert data1.shape == data2.shape
        
        #Want to resample from the distribution of paired differences with replacement
        paired_diff = data1 - data2
        data_len = paired_diff.shape[0]
        
        for i in nb.prange(M):
            results[i] = np.nanmean(rng.choice(paired_diff, size = data_len, replace = True))
    else:        
        data_len = data1.shape[0]
        
        #create a bucket with all data thrown inside
        pooled_data = np.empty(data1.shape[0] + data2.shape[0]) * np.nan
        pooled_data[:data1.shape[0]] = data1
        pooled_data[data1.shape[0]:] = data2
        
        #Recreate the two groups by sampling without replacement
        for i in nb.prange(M): 
            tmp = rng.choice(pooled_data, size = len(pooled_data), replace = False)
            data1_resample = tmp[:data_len] #up to number of points in data1
            data2_resample = tmp[data_len:] #the rest are in data2
            mean_diff = np.nanmean(data1_resample) - np.nanmean(data2_resample)
            results[i] = mean_diff
        
    #center the results on 0
    centered_results = results - np.nanmean(results)

    if alternative == "two-sided":
        #are the results more extreme than the original?
        p_val = np.sum(centered_results > abs(original_mean_diff)) + np.sum(centered_results < -abs(original_mean_diff))
        returned_distribution = centered_results
    elif alternative == "greater":
        #are results greater than the original?
        p_val = np.sum(centered_results - (original_mean_diff) > 0)
        returned_distribution = centered_results - abs(original_mean_diff)
    elif alternative == "less":
        #are results less than the original?
        p_val = np.sum(centered_results - (original_mean_diff) < 0)
        returned_distribution = centered_results + abs(original_mean_diff)
    else:
        raise ValueError("alternative must be \"two-sided\", \"greater\", or \"less\"")
        
    return p_val / M, returned_distribution
    

def Bootstrap(data1, data2, M = 1e4, paired = False, alternative = "two-sided", return_distribution = False, **kwargs):
    """ Bootstrap difference in means between two groups.
    M = float64 # Number of iterations
    paired = {True, False}
    alternative = {"two-sided", "greater", "less"} #data1 relative to data2, i.e.: data1 "greater" than data2. 
    return_distribution {True, False} #returns the bootstrapped distribution
    
    ##Legacy code for backwards compatability. Do not reccomend usage.
    direction = {"greater", "lesser", None} 
    if verbose = True, returns p_value and distribution
    if not verbose, only returns p-value
    """
    #BackCompat keywords
    direction = kwargs.get("direction", None)
    verbose = kwargs.get("verbose", False)
    
    #make sure data is in np.array format
    data1, data2 = np.array(data1), np.array(data2)
    
    #make M an integer
    M = int(M)
    
    #handle backwards compatability
    if direction == "greater": 
        alternative = direction
    elif direction == "lesser": 
        alternative = "less"
    elif direction != None: 
        raise ValueError("\"direction\" keyword misused. Please use the \"alternative\" keyward argument.")
    
    p_val, distribution = _NB_Bootstrap(data1, data2, M = M, paired = paired, alternative = alternative)
    
    if verbose or return_distribution:
        return p_val, distribution
    else:
        return p_val
    
@njit(parallel = True)
def _nb_mean_bootstrap(data1, data2, M = 1e4, paired = False, alternative = "two-sided", seed = None):
    rng = np.random

    results = np.empty(M) * np.nan
    original_mean_diff = np.nanmean(data1) - np.nanmean(data2)

    if paired:
        assert data1.shape == data2.shape
        
        #Want to resample from the distribution of paired differences with replacement
        paired_diff = data1 - data2
        data_len = paired_diff.shape[0]
        
        for i in nb.prange(M):
            if seed != None:
                rng.seed(int(i * seed))
            results[i] = np.nanmean(rng.choice(paired_diff, size = data_len, replace = True))
    else:        
        data_len = data1.shape[0]
        
        #create a bucket with all data thrown inside
        pooled_data = np.empty(data1.shape[0] + data2.shape[0]) * np.nan
        pooled_data[:data1.shape[0]] = data1
        pooled_data[data1.shape[0]:] = data2
        
        #Recreate the two groups by sampling without replacement
        for i in nb.prange(M): 
            if seed != None:
                rng.seed(int(i * seed))
            tmp = rng.choice(pooled_data, size = len(pooled_data), replace = False)
            data1_resample = tmp[:data_len] #up to number of points in data1
            data2_resample = tmp[data_len:] #the rest are in data2
            mean_diff = np.nanmean(data1_resample) - np.nanmean(data2_resample)
            results[i] = mean_diff
        
    #center the results on 0
    centered_results = results - np.nanmean(results)

    if alternative == "two-sided" or alternative == "two.sided":
        #are the results more extreme than the original?
        p_val = np.sum(centered_results > abs(original_mean_diff)) + np.sum(centered_results < -abs(original_mean_diff))
        returned_distribution = centered_results
    elif alternative == "greater":
        #are results greater than the original?
        p_val = np.sum(centered_results - (original_mean_diff) > 0)
        returned_distribution = centered_results - abs(original_mean_diff)
    elif alternative == "less":
        #are results less than the original?
        p_val = np.sum(centered_results - (original_mean_diff) < 0)
        returned_distribution = centered_results + abs(original_mean_diff)
    else:
        raise ValueError("alternative must be \"two-sided\", \"greater\", or \"less\"")
        
    return p_val / M, returned_distribution

@njit(parallel = True)
def _nb_median_bootstrap(data1, data2, M = 1e4, paired = False, alternative = "two-sided", seed = None):
    rng = np.random

    results = np.empty(M) * np.nan
    original_mean_diff = np.nanmedian(data1) - np.nanmedian(data2)

    if paired:
        assert data1.shape == data2.shape
        
        #Want to resample from the distribution of paired differences with replacement
        paired_diff = data1 - data2
        data_len = paired_diff.shape[0]
        
        for i in nb.prange(M):
            if seed != None:
                rng.seed(int(i * seed))
            results[i] = np.nanmedian(rng.choice(paired_diff, size = data_len, replace = True))
    else:        
        data_len = data1.shape[0]
        
        #create a bucket with all data thrown inside
        pooled_data = np.empty(data1.shape[0] + data2.shape[0]) * np.nan
        pooled_data[:data1.shape[0]] = data1
        pooled_data[data1.shape[0]:] = data2
        
        #Recreate the two groups by sampling without replacement
        for i in nb.prange(M): 
            if seed != None:
                rng.seed(int(i * seed))
            tmp = rng.choice(pooled_data, size = len(pooled_data), replace = False)
            data1_resample = tmp[:data_len] #up to number of points in data1
            data2_resample = tmp[data_len:] #the rest are in data2
            mean_diff = np.nanmedian(data1_resample) - np.nanmedian(data2_resample)
            results[i] = mean_diff
        
    #center the results on 0
    centered_results = results - np.nanmean(results)

    if alternative == "two-sided" or alternative == "two.sided":
        #are the results more extreme than the original?
        p_val = np.sum(centered_results > abs(original_mean_diff)) + np.sum(centered_results < -abs(original_mean_diff))
        returned_distribution = centered_results
    elif alternative == "greater":
        #are results greater than the original?
        p_val = np.sum(centered_results - (original_mean_diff) > 0)
        returned_distribution = centered_results - abs(original_mean_diff)
    elif alternative == "less":
        #are results less than the original?
        p_val = np.sum(centered_results - (original_mean_diff) < 0)
        returned_distribution = centered_results + abs(original_mean_diff)
    else:
        raise ValueError("alternative must be \"two-sided\", \"greater\", or \"less\"")
        
    return p_val / M, returned_distribution

def bootstrap(data1, data2, M = 1e4, paired = False, alternative = "two-sided", return_distribution = False, test = "mean", seed = None, **kwargs):
    """ 
    Bootstrap difference in means or medians between two groups.
    M = float64 # Number of iterations
    paired = {True, False}
    alternative = {"two-sided", "greater", "less"} #data1 relative to data2, i.e.: data1 "greater" than data2. "two.sided" is also accepted
    return_distribution {True, False} #returns the bootstrapped distribution
    test = {"mean", "median"} #compares either differences in means or median
    seed = int #modifies the seed used in the random number generator
    
    ##Legacy code for backwards compatability. Do not reccomend usage.
    direction = {"greater", "lesser", None} 
    if verbose = True, returns p_value and distribution
    if not verbose, only returns p-value
    """
    #BackCompat keywords
    direction = kwargs.get("direction", None)
    verbose = kwargs.get("verbose", False)
    
    #make sure data is in np.array format
    data1, data2 = np.array(data1), np.array(data2)
    
    #make M an integer
    M = int(M)
    
    #handle backwards compatability
    if direction == "greater": 
        alternative = direction
    elif direction == "lesser": 
        alternative = "less"
    elif direction != None: 
        raise ValueError("\"direction\" keyword misused. Please use the \"alternative\" keyward argument.")
    
    if test == "mean":
        p_val, distribution = _nb_mean_bootstrap(data1, data2, M = M, paired = paired, alternative = alternative, seed = seed)
    elif test == "median":
        p_val, distribution = _nb_median_bootstrap(data1, data2, M = M, paired = paired, alternative = alternative, seed = seed)
    else:
        raise ValueError("\"test\" must be \"mean\" or \"median\".")
        
    if verbose or return_distribution:
        return p_val, distribution
    else:
        return p_val
