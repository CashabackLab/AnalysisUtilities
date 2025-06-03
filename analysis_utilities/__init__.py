#allows using functions like analysis_utilities.custom_statistics.FunctionName()
#from . import custom_statistics
# from.custom_statistics import *

__version__ = "0.5.20"

from .stats.Bootstrap import bootstrap
from .stats.Grubbs import grubbs
from .stats.CLES import cles
from .stats.Cohen_D import cohen_d
from .stats.holmbonferroni_correction import holmbonferroni_correction

from .Linear_Fit import linear_fit
from .remove_nans import remove_nans