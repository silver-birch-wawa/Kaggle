import numpy as np
def rmse(data,compare_data):
    return np.sqrt(np.sum(np.sum(data-compare_data)**2))/np.sum(np.sum(data-compare_data!=0.0))