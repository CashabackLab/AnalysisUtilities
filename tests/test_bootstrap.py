import numpy as np
from analysis_utilities import compare_to_null
from analysis_utilities import bootstrap
from analysis_utilities import bootstrap_linear_regression
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

def test_nb_bootstrap_linear_regression_against_scipy():
    x_data_1 = np.random.normal(0,10,1000)
    y_data_1 = np.random.normal(0,10,1000)
    x_data_2 = np.random.normal(0,10,1000)
    y_data_2 = np.random.normal(0,10,1000)
    data_group_1 = np.array([x_data_1, y_data_1]).T #2d array for group 1
    data_group_2 = np.array([x_data_2, y_data_2]).T #2d array for group 2
    boot_dict = bootstrap_linear_regression(data_group_1, data_group_2, M = int(1e6), 
                                        paired = False, alternative = "two-sided")
    sp_slope_1 = float(sp.stats.linregress(x_data_1, y_data_1, alternative='two-sided').slope)
    sp_intercept_1 = float(sp.stats.linregress(x_data_1, y_data_1, alternative='two-sided').intercept)
    sp_slope_2 = float(sp.stats.linregress(x_data_2, y_data_2, alternative='two-sided').slope)
    sp_intercept_2 = float(sp.stats.linregress(x_data_2, y_data_2, alternative='two-sided').intercept)
    assert round(boot_dict['m_1'],3) == round(sp_slope_1, 3) and round(boot_dict['m_2'], 3) == round(sp_slope_2, 3) and round(boot_dict['b_1'], 3) == round(sp_intercept_1, 3) and round(boot_dict['b_2'], 3) == round(sp_intercept_2, 3) 

def test_nb_bootstrap_linear_regression_against_numpy():
    x_data_1 = np.random.normal(0,10,1000)
    y_data_1 = np.random.normal(0,10,1000)
    x_data_2 = np.random.normal(0,10,1000)
    y_data_2 = np.random.normal(0,10,1000)
    data_group_1 = np.array([x_data_1, y_data_1]).T #2d array for group 1
    data_group_2 = np.array([x_data_2, y_data_2]).T #2d array for group 2
    boot_dict = bootstrap_linear_regression(data_group_1, data_group_2, M = int(1e6), 
                                        paired = False, alternative = "two-sided")
    np_slope_1, np_intercept_1 = np.polyfit(x_data_1, y_data_1, 1)
    np_slope_2, np_intercept_2 = np.polyfit(x_data_2, y_data_2, 1)
    assert round(boot_dict['m_1'],3) == round(np_slope_1, 3) and round(boot_dict['m_2'], 3) == round(np_slope_2, 3) and round(boot_dict['b_1'], 3) == round(np_intercept_1, 3) and round(boot_dict['b_2'], 3) == round(np_intercept_2, 3)

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