import numpy as np

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

def CLES(data1, data2, paired = False, sample = "one sample", alt_comparison = 0):
    """
    Computes the common language effect size for the given data
    if sample = "one sample", data2 should be treated as a single number, mu, the mean to compare the data to
    if sample = "two sample", uses both inputs as data arrays
    if sample = "paired", computes paired difference then compares to 0 (can be changed using alt_comparison keyword)
    """
    if sample.lower() == "one sample" and paired == False:
        #treat data2 as mu for comparisons
        return Common_Language_EF_One_Sample(data1, data2)*100
    elif sample.lower() == "two sample" and paired == False:
        return Common_Language_EF_Two_Sample(data1, data2) *100
    elif sample.lower() == "paired" or paired == True:
        data = np.array(data1) - np.array(data2)
        return Common_Language_EF_One_Sample(data, alt_comparison) *100
