import numpy as np

def Cohen_D(data1, data2):
    data1, data2 = np.array(data1), np.array(data2)
    
    mean1, mean2 = np.nanmean(data1), np.nanmean(data2)
    std1, std2 = np.nanstd(data1), np.nanstd(data2)
    d = (mean1 - mean2) / ((std1**2 + std2**2)/2)**.5
    
    return abs(d)
