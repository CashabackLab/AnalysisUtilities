import numpy as np

def remove_nans(array):
    """
    Removes nan values from a numpy array or list
    Returns a flattened array with all nans removed
    """
    array = np.array(array)
    clean_array = array[~np.isnan(array)]
    
    return clean_array
