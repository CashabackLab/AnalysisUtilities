#allows using functions like analysis_utilities.custom_statistics.FunctionName()
#from . import custom_statistics
# from.custom_statistics import *

__version__ = "0.5.2"

from .stats.Bootstrap import Bootstrap, bootstrap
from .stats.Grubbs import Grubbs
from .stats.CLES import CLES, cles
from .stats.Cohen_D import Cohen_D
from .stats.holmbonferroni_correction import holmbonferroni_correction

from .Linear_Fit import Linear_Fit
from .remove_nans import remove_nans

#import .roth_utilities
