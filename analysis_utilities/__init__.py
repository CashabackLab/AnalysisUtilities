#allows using functions like analysis_utilities.custom_statistics.FunctionName()
#from . import custom_statistics
# from.custom_statistics import *

__version__ = "0.6.21"

from .stats.Bootstrap import compare_to_null
from .stats.Bootstrap import linear_regression_func
from .stats.Bootstrap import bootstrap
from .stats.Bootstrap import bootstrap_linear_regression
from .stats.Grubbs import grubbs
from .stats.CLES import cles
from .stats.Cohen_D import cohen_d
from .stats.holmbonferroni_correction import holmbonferroni_correction
from .stats.AIC_score import aic_score
from .stats.BIC_score import bic_score

from .Linear_Fit import linear_fit
from .remove_nans import remove_nans

