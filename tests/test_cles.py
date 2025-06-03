import numpy as np
import pytest
from analysis_utilities import cles


def test_cles_paired():
    test_data1 = np.array([1,2,3,4,5])
    test_data2 = test_data1 + 1
    
    # Test all greater
    cles_test1 = cles(test_data1, test_data2, paired = True, alternative = "greater")
    assert cles_test1 == 0.0
    
    # Test all less
    cles_test2 = cles(test_data1, test_data2, paired = True, alternative = "less")
    assert cles_test2 == 100.0
    
    test_data3 = np.array([2,1,2,3,4]) # Only one greater
    cles_test4 = cles(test_data1, test_data3, paired = True, alternative = "greater")
    assert cles_test4 == 80.0
    
    cles_test3 = cles(test_data1, test_data3, paired = True, alternative = "less")
    assert cles_test3 == 20.0
    
def test_cles_unpaired():
    test_data1 = np.array([1,2,3])
    test_data2 = np.array([4,5,6])
    
    cles_test1 = cles(test_data1, test_data2, paired = False, alternative = "greater")
    assert cles_test1 == 0.0
    
    cles_test1 = cles(test_data1, test_data2, paired = False, alternative = "less")
    assert cles_test1 == 100.0