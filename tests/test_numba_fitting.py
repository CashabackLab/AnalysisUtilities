# %%
import numpy as np
from analysis_utilities.utils import quadratic_func

def test_quadratic_fit0():
    x = np.arange(-5, 16, .001)
    a_coeff, b_coeff, c_coeff = -3, 2, 7
    y = a_coeff*x**2 + b_coeff*x + c_coeff

    a, b, c= quadratic_func(x, y)
    assert abs(a_coeff - a) < 0.0001 and abs(b_coeff - b) < 0.0001 and abs(c_coeff - c) < 0.0001

def test_quadratic_fit1():
    x = np.arange(5, 26, .001)
    a_coeff, b_coeff, c_coeff = 13, 2, -7
    y = a_coeff*x**2 + b_coeff*x + c_coeff

    a, b, c = quadratic_func(x, y)
    assert abs(a_coeff - a) < 0.0001 and abs(b_coeff - b) < 0.0001 and abs(c_coeff - c) < 0.0001

def test_quadratic_fit0():
    x = np.arange(5, 26, .001)
    a_coeff, b_coeff, c_coeff = 11, -11, -6
    y = a_coeff*x**2 + b_coeff*x + c_coeff

    a, b, c = quadratic_func(x, y)
    assert abs(a_coeff - a) < 0.0001 and abs(b_coeff - b) < 0.0001 and abs(c_coeff - c) < 0.0001