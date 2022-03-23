import numpy as np

def holmbonferroni_correction(p_value_array):
    """
    

    Parameters
    ----------
    p_value_array : numpy array
        1xN array of p-vals can be sorted or not sorted.

    Returns
    -------
    corrected_p_vals: numpy array
        Returns holm-bonferonnicorrection in original array order

    """
    
    num_pvals = len(p_value_array)
    sorted_pvals = np.sort(p_value_array,0)    
    sorted_args_pvals = np.argsort(p_value_array,0) 
    corrected_p_vals = np.zeros_like(sorted_pvals)*np.nan   
    
    for i in range(num_pvals):
        corrected_p_vals[sorted_args_pvals[i]] = sorted_pvals[i] * (num_pvals-i)
        
    return corrected_p_vals
