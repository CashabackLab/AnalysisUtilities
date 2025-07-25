import numpy as np
import warnings

def _common_language_ef_one_sample(data, mu):
    total = 0
    for x in data:
        if x > mu:
            total+= 1
        elif x == mu:
            total += .5
        else:
            total += 0
    return total / len(data)

def _common_language_ef_two_sample(data1, data2):
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
    
def cles(data1, data2 = 0, paired = False, alternative = "greater"):
    """
    Computes the common language effect size for the given data.
    If only one data array is given, computes one sample cles
    If only one data array is given and second input is a number, 
        computes cles comparison to the given number
    alternative : {greater, less}; default is greater
    paired: {True, False}, must give data array as second input
    normalize: {True, False}, normalize result to be between 50 - 100
    """    
    array_flag = 0
    if type(data2) == type(list()) or type(data2) == type(np.array(np.nan)):
        array_flag = 1
        data2 = np.array(data2)
        
    data1 = np.array(data1)
    ##if two samples given and not paired
    if alternative == "greater" and paired == False and array_flag:
        theta = _common_language_ef_two_sample(data1, data2)
        
    elif alternative == "less" and paired == False and array_flag:
        theta = _common_language_ef_two_sample(data2, data1)
        
    ##if one sample is given and second sample is value to compare to
    elif alternative == "greater" and paired == False and not array_flag:
        theta = _common_language_ef_two_sample(data1, data2)
        
    elif alternative == "less" and paired == False and not array_flag:
        theta = _common_language_ef_two_sample(data2, data1)
    
    ## if two samples are given and they are paired
    elif alternative == "greater" and paired == True:
        theta = _common_language_ef_one_sample(data1 - data2, 0)
        
    elif alternative == "less" and paired == True:
        theta = _common_language_ef_one_sample(data2 - data1, 0)
    else: 
        raise ValueError(f"Invalid alternative argument: \"{alternative}\". Valid Arguments are: [greater, less]") 
            
    return theta * 100
