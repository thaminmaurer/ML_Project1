import numpy as np
import matplotlib.pyplot as plt


### standardize
def standardize(x):
    """Standardize the original data set."""
    mean_x = np.mean(x, axis=0)
    print(np.shape(mean_x))
    x = x - mean_x
    std_x = np.std(x, axis=0)
    print(np.shape(std_x))
    x = x / (std_x+1e-12) ### the small term is to deal with the division by zero 
    return x, mean_x, std_x





