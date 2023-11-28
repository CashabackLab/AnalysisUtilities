import numpy as np
import warnings

def _Common_Language_EF_One_Sample(data, mu):
    total = 0
    for x in data:
        if x > mu:
            total+= 1
        elif x == mu:
            total += .5
        else:
            total += 0
    return total / len(data)

def _Common_Language_EF_Two_Sample(data1, data2):
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

def CLES(data1, data2, paired = False, sample = "two sample", alt_comparison = 0):
    warnings.warn("Outdated function, please use \cles\ for better compatability.", DeprecationWarning)
    """
    Computes the common language effect size for the given data
    if sample = "one sample", data2 should be treated as a single number, mu, the mean to compare the data to
    if sample = "two sample", uses both inputs as data arrays
    if sample = "paired", computes paired difference then compares to 0 (can be changed using alt_comparison keyword)
    """
    if sample.lower() == "one sample" and paired == False:
        #treat data2 as mu for comparisons
        return _Common_Language_EF_One_Sample(data1, data2)*100
    elif sample.lower() == "two sample" and paired == False:
        return _Common_Language_EF_Two_Sample(data1, data2) *100
    elif sample.lower() == "paired" or paired == True:
        data = np.array(data1) - np.array(data2)
        return _Common_Language_EF_One_Sample(data, alt_comparison) *100
    
def cles(data1, data2 = 0, paired = False, alternative = "greater", normalize = False, **kwargs):
    """
    Computes the common language effect size for the given data.
    If only one data array is given, computes one sample cles
    If only one data array is given and second input is a number, 
        computes cles comparison to the given number
    alternative : {greater, less}; default is greater
    paired: {True, False}, must give data array as second input
    normalize: {True, False}, normalize result to be between 50 - 100
    """
    kwargs.get("normalize", False)
    
    array_flag = 0
    if type(data2) == type(list()) or type(data2) == type(np.array(np.nan)):
        array_flag = 1
        data2 = np.array(data2)
        
    data1 = np.array(data1)
    ##if two samples given and not paired
    if alternative == "greater" and paired == False and array_flag:
        theta = _Common_Language_EF_Two_Sample(data1, data2)
        
    elif alternative == "less" and paired == False and array_flag:
        theta = _Common_Language_EF_Two_Sample(data2, data1)
        
    ##if one sample is given and second sample is value to compare to
    elif alternative == "greater" and paired == False and not array_flag:
        theta = _Common_Language_EF_Two_Sample(data1, data2)
        
    elif alternative == "less" and paired == False and not array_flag:
        theta = _Common_Language_EF_Two_Sample(data2, data1)
    
    ## if two samples are given and they are paired
    elif alternative == "greater" and paired == True:
        theta = _Common_Language_EF_One_Sample(data1 - data2, 0)
        
    elif alternative == "less" and paired == True:
        theta = _Common_Language_EF_One_Sample(data2 - data1, 0)
    else: 
        raise ValueError(f"Invalid alternative argument: \"{alternative}\". Valid Arguments are: [greater, less]") 

    if normalize:
        if theta < .5:
            theta = 1-theta
            
    return theta * 100
