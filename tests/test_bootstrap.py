import pytest
import sys
from analysis_utilities import bootstrap

def test_nb_bootstrap():
    fake_data = 1
    test_data = 1
    # assert fake_data != test_data

def test_add():
    my_output = 5
    function_output = bootstrap(2,3)
    assert my_output == function_output
    
if __name__ == "__main__":
    # test_nb_bootstrap()
    test_add()
    