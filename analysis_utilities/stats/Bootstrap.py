import numpy as np
import numba as nb
from numba.core.errors import NumbaDeprecationWarning, NumbaPendingDeprecationWarning
import warnings
# import sklearn.linear_model as lm
from analysis_utilities.utils import nb_nanmean, nb_nanmedian

warnings.simplefilter('ignore', category=NumbaDeprecationWarning)
warnings.simplefilter('ignore', category=NumbaPendingDeprecationWarning)

@nb.jit
def compare_to_null(results_distribution, original_value_diff, alternative):

    if alternative == "two-sided":
        #are the results as extreme or more extreme than the original?        
        p_val = np.sum(np.abs(results_distribution) - np.abs(original_value_diff)>=0)
        # returned_distribution = results_distribution
    elif alternative == "greater":
        #are results greater than or equal to the original?
        p_val = np.sum((results_distribution - original_value_diff) >= 0)
        # returned_distribution = results_distribution - abs(original_value_diff)
    elif alternative == "less":
        #are results less than or equal to the original?
        p_val = np.sum((results_distribution - original_value_diff) <= 0)
        # returned_distribution = results_distribution + abs(original_value_diff)
    else:
        raise ValueError("alternative must be \"two-sided\", \"greater\", or \"less\"")
    
    final_pval = p_val / len(results_distribution)

    return final_pval


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
            data_group_1_resample = tmp_data[:data_len] #up to number of points in data1
            data_group_2_resample = tmp_data[data_len:] #the rest are in data2
            mean_diff = avg_function(data_group_1_resample, axis=0) - avg_function(data_group_2_resample, axis=0)
            results[i] = mean_diff
        
    # Center the results on 0; technically don't need to do this for between (only paired), since
    # it will already be centered on zero (given enough bootstraps)
    centered_results = results - np.nanmean(results)

    # Get original mean diff
    original_mean_diff = avg_function(data1, axis=0) - avg_function(data2, axis=0)

    final_pval = compare_to_null(centered_results, original_mean_diff, alternative)

    return final_pval, centered_results

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

@nb.jit
def linear_regression_func(x, y):
    data_x = nb_nanmean(x, axis=0)
    data_y = nb_nanmean(y, axis=0)
    n = len(data_x)
    # https://www.geeksforgeeks.org/maths/linear-regression-formula/
    sum_x = np.sum(data_x)
    sum_y = np.sum(data_y)
    sum_xy = np.sum(data_x*data_y)
    sum_x_squared = np.sum(data_x**2)

    denominator = (n*sum_x_squared) - sum_x**2

    if denominator == 0:
        raise ValueError("denominator calculation is 0, denominator set to nan")
        # return np.nan, np.nan
        # denominator = np.nan

    numerator_m = (n*sum_xy) - (sum_x*sum_y)
    numerator_b = (sum_y * sum_x_squared) - (sum_x * sum_xy)

    m = numerator_m/denominator
    b = numerator_b/denominator
    return m, b

@nb.jit
def sigmoid_func(data_x, data_y_temp):
    data_y = -np.log((1/data_y_temp)-1)
    return linear_regression_func(data_x, data_y)

@nb.njit(parallel = True)
def _nb_bootstrap_slopes(
    data_group_1, data_group_2,
    calc_function, 
    M:int, 
    paired:bool, 
    alternative:str, 
    seed:int = None
): 
    
    # create empty arrays to store results
    results_store_slopes = np.empty(M) * np.nan
    results_store_intercepts = np.empty(M) * np.nan
    group_1_resampled_slopes = np.empty(M) * np.nan
    group_2_resampled_slopes = np.empty(M) * np.nan
    group_1_resampled_intercepts = np.empty(M) * np.nan
    group_2_resampled_intercepts = np.empty(M) * np.nan
    
    # create a bucket with all data thrown inside
    pooled_data = np.concatenate((data_group_1, data_group_2), axis=1)
    n_pooled = np.shape(pooled_data)[1]
    data_len = int(n_pooled/2)
    
    # Recreate the two groups by sampling without replacement
    for i in nb.prange(M): 
    # for i in range(M): 
        if seed != None:
            np.random.seed(int((i+1) * seed))
        # populate a randomized list of indices to resample the two groups
        reordered_list_idx = np.random.choice(n_pooled, size=n_pooled, replace = False)
        # reorder the bucket of groups, keeping their respective x and y values together
        resampled_data = pooled_data[:,reordered_list_idx]
        
        # separate the resampled data into two groups
        data_group_1_resample = resampled_data[:,:data_len] #up to number of points in data1
        data_group_2_resample = resampled_data[:,data_len:] #the rest are in data2

        # calculate slope and y-intercept using the resampled data
        m_1, b_1 = calc_function(data_group_1_resample[:,:][0], data_group_1_resample[:,:][1])
        m_2, b_2 = calc_function(data_group_2_resample[:,:][0], data_group_2_resample[:,:][1])

        # store the slope and y-intercept differences between groups
        group_1_resampled_slopes[i] = m_1
        group_2_resampled_slopes[i] = m_2
        group_1_resampled_intercepts[i] = b_1
        group_2_resampled_intercepts[i] = b_2
        results_store_slopes[i] = m_1 - m_2
        results_store_intercepts[i] = b_1 - b_2

    #get original slope and intercept differences
    original_m_1, original_b_1 = calc_function(data_group_1[:,:][0], data_group_1[:,:][1])
    original_m_2, original_b_2 = calc_function(data_group_2[:,:][0], data_group_2[:,:][1])
    # differences of the original data's slopes and y-intercepts
    original_m_diff = original_m_1 - original_m_2
    original_b_diff = original_b_1 - original_b_2

    # Center the results on 0; technically don't need to do this for between (only paired), since
    # it will already be centered on zero (given enough bootstraps)
    centered_results_m = results_store_slopes - np.nanmean(results_store_slopes)
    centered_results_b = results_store_intercepts - np.nanmean(results_store_intercepts)

    final_pval_m = compare_to_null(centered_results_m, original_m_diff, alternative)
    final_pval_b = compare_to_null(centered_results_b, original_b_diff, alternative)
    
    return final_pval_m, centered_results_m, final_pval_b, centered_results_b, group_1_resampled_slopes, group_2_resampled_slopes, group_1_resampled_intercepts, group_2_resampled_intercepts


def bootstrap_linear_regression(data_group_1, data_group_2, calc_function=linear_regression_func,
              M:int = int(1e4), 
              paired:bool = False, 
              alternative:str = "two-sided",  
              seed:int = None):
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
    
    # Check sample sizes
    if paired:
        assert data_group_1.shape == data_group_2.shape
    else:        
        if data_group_1.shape != data_group_2.shape:
            warnings.warn("Sample sizes not the same (data_group_1 shape and data_group_2 shape are not the same).", UserWarning)
            
    final_pval_m, returned_distribution_m, final_pval_b, returned_distribution_b, group_1_resampled_slopes, group_2_resampled_slopes, group_1_resampled_intercepts, group_2_resampled_intercepts = _nb_bootstrap_slopes(data_group_1, data_group_2, 
                                                                                    calc_function, M = M, 
                                                                                    paired = paired, 
                                                                                    alternative = alternative, 
                                                                                    seed = seed)
    return dict({'slope_diff_pval': final_pval_m, 'intercept_diff_pval': final_pval_b,
                 'slope_difference_distribution': returned_distribution_m,
                 'intercept_difference_distribution': returned_distribution_b,
                 'resampled_slope_group1_distribution': group_1_resampled_slopes,
                 'resampled_slope_group2_distribution': group_2_resampled_slopes,
                 'resampled_intercept_group1_distribution': group_1_resampled_intercepts,
                 'resampled_intercept_group2_distribution': group_2_resampled_intercepts})

def bootstrap_sigmoid(data_group_1, data_group_2, calc_function=sigmoid_func,
              M:int = int(1e4), 
              paired:bool = False, 
              alternative:str = "two-sided",  
              seed:int = None):
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
    
    # Check sample sizes
    if paired:
        assert data_group_1.shape == data_group_2.shape
    else:        
        if data_group_1.shape != data_group_2.shape:
            warnings.warn("Sample sizes not the same (data_group_1 shape and data_group_2 shape are not the same).", UserWarning)
            
    final_pval_m, returned_distribution_m, final_pval_b, returned_distribution_b, group_1_resampled_slopes, group_2_resampled_slopes, group_1_resampled_intercepts, group_2_resampled_intercepts = _nb_bootstrap_slopes(data_group_1, data_group_2, 
                                                                                    calc_function, M = M, 
                                                                                    paired = paired, 
                                                                                    alternative = alternative, 
                                                                                    seed = seed)
    return dict({'slope_diff_pval': final_pval_m, 'intercept_diff_pval': final_pval_b,
                 'slope_difference_distribution': returned_distribution_m,
                 'intercept_difference_distribution': returned_distribution_b,
                 'resampled_slope_group1_distribution': group_1_resampled_slopes,
                 'resampled_slope_group2_distribution': group_2_resampled_slopes,
                 'resampled_intercept_group1_distribution': group_1_resampled_intercepts,
                 'resampled_intercept_group2_distribution': group_2_resampled_intercepts})