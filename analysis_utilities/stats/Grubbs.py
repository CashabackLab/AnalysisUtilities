import numpy as np
from scipy import stats

def grubbs(data, type = "Two Sided", alpha = 0.05, verbose = False):
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
