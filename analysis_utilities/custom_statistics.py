# -*- coding: utf-8 -*-
"""
Created on Mon Sep 27 12:58:31 2021

@author: adam1
"""
#from Custom_Package import custom_stats_package as csp

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

def Two_Tailed_Bootstrap(data1, data2, M = 1e4, paired = False, verbose = False):
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
def NB_Two_Tailed_Bootstrap(data1, data2, M = 1e4, paired = False, verbose = False):
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

def One_Tailed_Bootstrap(data1, data2, M = 1e4, paired = False, direction = "lesser", verbose = False):
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
def NB_One_Tailed_Bootstrap(data1, data2, M = 1e4, paired = False, direction = "lesser", verbose = False):
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
            return NB_Two_Tailed_Bootstrap(data1, data2, M = M, paired = paired, verbose = verbose)
        else:
            return NB_One_Tailed_Bootstrap(data1, data2, M = M, paired = paired, direction = direction,  verbose = verbose)
    else:
        if direction == None or direction == "two-tailed":
            return Two_Tailed_Bootstrap(data1, data2, M = M, paired = paired, verbose = verbose)
        else:
            return One_Tailed_Bootstrap(data1, data2, M = M, paired = paired, direction = direction,  verbose = verbose)

def Grubbs(data, type = "Two Sided", alpha = 0.05, verbose = False):
    '''
    Performs the Grub's Outlier test on data.
    Set verbose to true to see the grubbs value and threshold
    Returns NaN if no outlier is present
    Else returns the index of the outlier
    '''
    data = np.array(data)
    mean = np.nanmean(data)
    std = np.std(data)
    dof = data.shape[0] -1
    N = data.shape[0]
    
    if type.lower() == "two sided":
        t = stats.t.ppf(q = 1 - alpha / (2*N), df = dof)
        G = np.max(np.abs(data - mean)) / std
        threshold = (N-1) / N**.5 * (t**2 / (N-2 + t**2))**.5
        if verbose:
            print(f"G = {G:.4f}\nGrubbs Threshold = {threshold:.4f}")
        if G > threshold:
            return np.argmax(np.abs(data - mean))
        else:
            return np.nan
        
    elif type.lower() == "greater":
        t = stats.t.ppf(q = 1 - alpha / (N), df = dof)
        G = (np.max(data) - mean) / std
        threshold = (N-1) / N**.5 * (t**2 / (N-2 + t**2))**.5
        if verbose:
            print(f"G = {G:.4f}\nGrubbs Threshold = {threshold:.4f}")
        if G > threshold:
            return np.argmax(data)
        else:
            return np.nan
        
    elif type.lower() == "lesser":
        t = stats.t.ppf(q = 1 - alpha / (N), df = dof)
        G = (mean - np.min(data)) / std
        threshold = (N-1) / N**.5 * (t**2 / (N-2 + t**2))**.5
        if verbose:
            print(f"G = {G:.4f}\nGrubbs Threshold = {threshold:.4f}")
        if G > threshold:
            return np.argmin(data)
        else:
            return np.nan
        
def Linear_Fit(array_A, array_B):
    """
    Returns slope and y-intercept of the line of best fit
    """
    array_A = np.array(array_A)
    array_B = np.array(array_B)
    
    #Pair arrays then sort them for easier fit
    zipped_list = zip(array_A[~np.isnan(array_A)], array_B[~np.isnan(array_B)])
    sorted_list = sorted(zipped_list)
    sorted_a, sorted_b = zip(*sorted_list)
    
    m, b = np.polyfit(sorted_a, sorted_b, 1)
    return m, b

def Cohen_D(data1, data2):
    data1, data2 = np.array(data1), np.array(data2)
    
    mean1, mean2 = np.nanmean(data1), np.nanmean(data2)
    std1, std2 = np.nanstd(data1), np.nanstd(data2)
    d = (mean1 - mean2) / ((std1**2 + std2**2)/2)**.5
    
    return abs(d)

def Common_Language_EF_One_Sample(data, mu):
    total = 0
    for x in data:
        if x > mu:
            total+= 1
        elif x == mu:
            total += .5
        else:
            total += 0
    return total / len(data)

def Common_Language_EF_Two_Sample(data1, data2):
    total = 0
    for x in data1:
        for y in data2:
            if x > y:
                total += 1
            elif x == y:
                total += .5
            else:
                total += 0
    return total / (len(data1) * len(data2))

def CLES(data1, data2, sample = "one sample", alt_comparison = 0):
    """
    Computes the common language effect size for the given data
    if sample = "one sample", data2 should be treated as a single number, mu, the mean to compare the data to
    if sample = "two sample", uses both inputs as data arrays
    if sample = "paired", computes paired difference then compares to 0 (can be changed using alt_comparison keyword)
    """
    if sample.lower() == "one sample":
        #treat data2 as mu for comparisons
        return Common_Language_EF_One_Sample(data1, data2)*100
    elif sample.lower() == "two sample":
        return Common_Language_EF_Two_Sample(data1, data2) *100
    elif sample.lower() == "paired":
        data = np.array(data1) - np.array(data2)
        return Common_Language_EF_One_Sample(data, alt_comparison) *100
