import numpy as np

def remove_nans(array, replace_value = None):
    """
    Removes nan values from a numpy array or list
    Returns a flattened array with all nans removed by default
    set replace_value to a value to maintain original dimensions and replace nans with the specified value
    """
    array = np.array(array)
    if replace_value == None:
        clean_array = array[~np.isnan(array)]
    else:
        clean_array = np.where(np.isfinite(array), array, replace_value)

    return clean_array
