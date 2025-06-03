import numpy as np

def linear_fit(array_A, array_B):
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
