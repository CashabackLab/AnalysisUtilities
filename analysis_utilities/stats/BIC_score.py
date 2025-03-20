import numpy as np

def bic_score(num_parameters, num_data_points, mean_squared_error):
    '''Calculates the BIC (Bayesian Information Criterion) of our model, assuming our model fit residuals are distributed as a gaussian. This is a fair assumption if the data is gaussian and your model fits reasonably well to the mean of the data. Uses the mean squared error of the model fit to the data.
    '''
    return num_parameters*np.log(num_data_points) + num_data_points*np.log(mean_squared_error)
