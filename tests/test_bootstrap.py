# %%
import numpy as np
from analysis_utilities import compare_to_null
from analysis_utilities import linear_regression_func
from analysis_utilities import sigmoid_func
from analysis_utilities import bootstrap
import pingouin as pg
import scipy as sp

alternatives = ["two-sided", "greater", "less"]

def test_nb_bootstrap_edgecases_paired():
    
    data1 = np.array([1,2,3])
    data2 = np.array([4,5,6])
    expected_pval = [0.0, 1.0, 0.0]
    
    for i,alternative in enumerate(alternatives):
        for j,stat_type in enumerate(["mean", "median"]):
            out = bootstrap(data1, data2, 
                            M = int(1e6), 
                            paired=True, 
                            alternative = alternative,  
                            stat_type = stat_type, 
                            return_distribution = False, 
                            seed=10)
    
            assert out == expected_pval[i], f"{i} didn't work"

def test_nb_boostrap_unpaired():
    assert True

def test_nb_boostrap_against_pingouin():
    
   data1 = np.random.normal(0,10,1000) 
   data2_means = [0,10,15]
   for i,alternative in enumerate(alternatives):
        for j,paired in enumerate([True, False]):
            for k, data2_mean in enumerate(data2_means):
                data2 = np.random.normal(data2_mean,10,1000) 
                ttest_pval = pg.ttest(data1, data2, paired = paired, alternative = alternative)["p-val"].item()
                boot_pval = bootstrap(data1, data2, 
                                M = int(1e6), 
                                paired=paired, 
                                alternative = alternative,  
                                stat_type = "mean", 
                                return_distribution = False, 
                                seed=10)
                print(ttest_pval, boot_pval)
                assert np.abs(ttest_pval - boot_pval) < 0.01
            
def test_nb_bootstrap_josh_boot_example():
    CORRECT_OUTPUT = 0.0262
    g_control = [87, 90, 82, 77, 71, 81, 77, 79, 84, 86, 78, 84, 86, 69, 81, 75, 70, 76, 75, 93]
    g_drug = [74, 67, 81, 61, 64, 75, 81, 81, 81, 67, 72, 78, 83, 85, 56, 78, 77, 80, 79, 74]
    boot_pval = bootstrap(g_control, g_drug, 
                                M = int(1e6), 
                                paired=False, 
                                alternative = "two-sided",  
                                stat_type = "mean", 
                                return_distribution = False, 
                                seed=20)
    assert np.abs(boot_pval - CORRECT_OUTPUT) < 0.005 

def test_nb_bootstrap_linear_regression_against_numpy():
    m_1, b_1 = -20, 3.14

    x_data_1 = np.array([np.arange(0,21).tolist()]*5)
    y_data_1 = (m_1 * x_data_1) + b_1

    test_m, test_b = linear_regression_func(x_data_1, y_data_1)
    np_slope, np_intercept = np.polyfit(x_data_1.ravel(), y_data_1.ravel(), 1)
    assert abs(test_m - np_slope) < 0.0001 and abs(test_b - np_intercept) < 0.0001

def test_nb_bootstrap_linear_regression_against_scipy():
    m_1, b_1 = 10, 77

    x_data_1 = np.array([np.arange(0,21).tolist()]*5)
    y_data_1 = (m_1 * x_data_1) + b_1

    test_m, test_b = linear_regression_func(x_data_1, y_data_1)
    sp_slope = float(sp.stats.linregress(x_data_1.ravel(), y_data_1.ravel(), alternative='two-sided').slope)
    sp_intercept = float(sp.stats.linregress(x_data_1.ravel(), y_data_1.ravel(), alternative='two-sided').intercept)
    
    assert abs(test_m - sp_slope) < 0.0001 and abs(test_b - sp_intercept) < 0.0001

def test_sigmoid_fit0():
    known_weight, known_bias = 10, -5

    x = np.array([np.arange(0, 1, .001).tolist()]*5)
    known_sigmoid = 1 / (1 + np.exp(-(known_weight * x + known_bias))) 

    test_m, test_b = sigmoid_func(x, known_sigmoid)
    assert abs(test_m - known_weight) < 0.0001 and abs(test_b - known_bias) < 0.0001

def test_sigmoid_fit1():
    known_weight, known_bias = -10, -5

    x = np.array([np.arange(0, 1, .001).tolist()]*5)
    known_sigmoid = 1 / (1 + np.exp(-(known_weight * x + known_bias))) 

    test_m, test_b = sigmoid_func(x, known_sigmoid)
    assert abs(test_m - known_weight) < 0.0001 and abs(test_b - known_bias) < 0.0001

def test_sigmoid_fit3():
    known_weight, known_bias = -10, 5

    x = np.array([np.arange(0, 1, .001).tolist()]*5)
    known_sigmoid = 1 / (1 + np.exp(-(known_weight * x + known_bias))) 

    test_m, test_b = sigmoid_func(x, known_sigmoid)
    assert abs(test_m - known_weight) < 0.0001 and abs(test_b - known_bias) < 0.0001

def test_sigmoid_fit4():
    known_weight, known_bias = -1, 5

    x = np.array([np.arange(-5, 16, .001).tolist()]*5)
    known_sigmoid = 1 / (1 + np.exp(-(known_weight * x + known_bias))) 

    test_m, test_b = sigmoid_func(x, known_sigmoid)
    assert abs(test_m - known_weight) < 0.0001 and abs(test_b - known_bias) < 0.0001

def test_compare_to_null_numbers_twosided():
    distribution = np.arange(-5, 11, 1)
    diff = 7
    alt = "two-sided"
    pval = compare_to_null(distribution, diff, alternative=alt)
    correct_pval = .25
    assert pval == correct_pval

def test_compare_to_null_norm_twosided():
    distribution = np.random.normal(0,10,1000)
    diff = 10
    alt = "two-sided"
    pval = compare_to_null(distribution, diff, alternative=alt)
    correct_pval = np.sum(np.abs(distribution) >= np.abs(diff))/len(distribution)
    assert (pval == correct_pval).item()

def test_compare_to_null_unniform_twosided():
    distribution = np.random.normal(0,10,1000)
    diff = 5
    alt = "two-sided"
    pval = compare_to_null(distribution, diff, alternative=alt)
    correct_pval = np.sum(np.abs(distribution) >= np.abs(diff))/len(distribution)
    assert (pval == correct_pval).item()

def test_compare_to_null_unniform_greater():
    distribution = np.random.uniform(0,10,1000)
    diff = 5
    alt = "greater"
    pval = compare_to_null(distribution, diff, alternative=alt)
    correct_pval = np.sum(distribution >= diff)/len(distribution)
    assert (pval == correct_pval).item()

def test_compare_to_null_norm_less():
    distribution = np.random.normal(0,10,1000)
    diff = 5
    alt = "less"
    pval = compare_to_null(distribution, diff, alternative=alt)
    correct_pval = np.sum(distribution <= diff)/len(distribution)
    assert (pval == correct_pval).item()