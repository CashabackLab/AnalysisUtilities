import numpy as np
import numba as nb
from numba import njit

@njit(parallel = True, fastmath = False) # nb.prange will only run if parallel = True
def aic_score(data, estimates, num_params):
    '''Calculates the AIC (Akaike Information Criterion) of our model, assuming our 
    model fit residuals are distributed as a gaussian. This is a fair assumption if 
    the data is gaussian and your model fits reasonably well to the mean of the data.
    Requires that the model was fit to minimize the mean squared error of the residuals.

    Derivation is shown in Cashaback Drive, under modeling resources.

    data, N_ixM (potentially) jagged list, the data the model was fit to. N_i datapoints for each of 
        M experimental conditions.
    estimates, 1D array, contains the best estimates (that minimize the mean squared error) for each M 
        experimental condition in the same respective order as the M experimental condition order for data.
    num_params, integer, the number of model fit parameters in the experimental model
    '''

    # Calculating MSE
    MSE = 0
    num_pts = 0
    for i in nb.prange(len(data)):
        num_pts += len(data[i])
        MSE_i = np.mean((data[i] - estimates[i])**2)
        MSE += MSE_i

    return 2*num_params + num_pts*np.log(MSE)
