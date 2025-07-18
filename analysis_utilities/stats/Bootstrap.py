import numpy as np
import numba as nb
from numba.core.errors import NumbaDeprecationWarning, NumbaPendingDeprecationWarning
import warnings
from analysis_utilities.utils import nb_nanmean, nb_nanmedian

warnings.simplefilter('ignore', category=NumbaDeprecationWarning)
warnings.simplefilter('ignore', category=NumbaPendingDeprecationWarning)
    
@nb.njit(parallel = True)
def _nb_bootstrap(
    data1, data2, 
    avg_function,
    M:int, 
    paired:bool, 
    alternative:str, 
    seed:int = None
): 
    
    rng = np.random
    M = int(M)
    results = np.empty(M) * np.nan

    if paired:        
        #Want to resample from the distribution of paired differences with replacement
        paired_diff = data1 - data2
        data_len = paired_diff.shape[0]
        
        for i in nb.prange(M):
            if seed != None:
                rng.seed(int(i * seed))
            results[i] = avg_function(rng.choice(paired_diff, size = data_len, replace = True), axis=0)
    else:        
        data_len = data1.shape[0]
        
        #create a bucket with all data thrown inside
        pooled_data = np.concatenate((data1,data2), axis=0)
        
        #Recreate the two groups by sampling without replacement
        for i in nb.prange(M): 
            if seed != None:
                rng.seed(int(i * seed))
            tmp_data = rng.choice(pooled_data, size = len(pooled_data), replace = False)
            data1_resample = tmp_data[:data_len] #up to number of points in data1
            data2_resample = tmp_data[data_len:] #the rest are in data2
            mean_diff = avg_function(data1_resample, axis=0) - avg_function(data2_resample, axis=0)
            results[i] = mean_diff
        
    # Center the results on 0; technically don't need to do this for between (only paired), since
    # it will already be centered on zero (given enough bootstraps)
    centered_results = results - np.nanmean(results)

    # Get original mean diff
    original_mean_diff = avg_function(data1, axis=0) - avg_function(data2, axis=0)

    if alternative == "two-sided":
        #are the results as extreme or more extreme than the original?        
        p_val = np.sum(np.abs(centered_results) - np.abs(original_mean_diff)>=0)
        returned_distribution = centered_results
    elif alternative == "greater":
        #are results greater than or equal to the original?
        p_val = np.sum((centered_results - original_mean_diff) >= 0)
        returned_distribution = centered_results - abs(original_mean_diff)
    elif alternative == "less":
        #are results less than or equal to the original?
        p_val = np.sum((centered_results - original_mean_diff) <= 0)
        returned_distribution = centered_results + abs(original_mean_diff)
    else:
        raise ValueError("alternative must be \"two-sided\", \"greater\", or \"less\"")
    
    final_pval = p_val / M
    
    return final_pval, returned_distribution

def bootstrap(data1, data2, 
              M:int = int(1e4), 
              paired:bool = False, 
              alternative:str = "two-sided",  
              stat_type:str = "mean", 
              return_distribution:bool = False, 
              seed:int = None, 
              **kwargs):
    ''' Performs a bootstrapped permutation test on our data. For a between test, will take two sets of 
    data and pool them. Two groups are then randomly sampled repetitively from this pool with replacement 
    and a mean difference is calculated for each set of two groups sampled. This will then produce M 
    bootstraps of our statistic (mean difference) and a numerical null distribution. The p-value is the 
    proportion of the data more extreme (either one or two sided) than our original data samples test 
    statistic (mean difference). 
    
    A paired test will resample from the pool of mean differences, and then center that distribution on 
    zero given the assumption of the null (no mean difference).
    
    In order to test a data set against zero, paired == True should be selected, and the second data set 
    will be zeros of the same length as the first data set. This will generate a null distribution for   
    just the first data set.
    
    M = float64 # Number of iterations
    paired = {True, False}
    alternative = {"two-sided", "greater", "less"} #data1 relative to data2, i.e.: data1 "greater" than 
    data2. "two.sided" is also accepted
    return_distribution {True, False} #returns the bootstrapped distribution
    test = {"mean", "median"} #compares either differences in means or median
    seed = int #modifies the seed used in the random number generator
    '''
    
    # Select averaging function
    if stat_type == "mean":
        avg_function = nb_nanmean
    elif stat_type == "median":
        avg_function = nb_nanmedian
    else:
        raise ValueError("stat_type should be 'mean' or 'median'")    
    
    assert alternative in ["two-sided", "greater", "less"]
    #make sure data is in np.array format
    data1, data2 = np.array(data1), np.array(data2)
    
    #make M an integer
    M = int(M)
    
    # Check sample sizes
    if paired:
        assert data1.shape == data2.shape
    else:        
        if data1.shape != data2.shape:
            warnings.warn("Sample sizes not the same (data1 shape and data2 shape are not the same).", UserWarning)
            
    p_val, distribution = _nb_bootstrap(data1, data2, 
                                        avg_function=avg_function, 
                                        M = M, 
                                        paired = paired, 
                                        alternative = alternative, 
                                        seed = seed)

        
    if return_distribution:
        return p_val, distribution
    else:
        return p_val
