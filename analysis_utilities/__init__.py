#allows using functions like analysis_utilities.custom_statistics.FunctionName()
#from . import custom_statistics
# from.custom_statistics import *

from .Bootstrap import Bootstrap
from .Grubbs import *
from .Linear_Fit import *
from .CLES import *
from .Cohen_D import *
from .holmbonferroni_correction import *
from .remove_nans import remove_nans
__version__ = "0.4.2"
