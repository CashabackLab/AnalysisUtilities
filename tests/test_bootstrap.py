import numpy as np
from analysis_utilities import bootstrap
import pingouin as pg

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

    