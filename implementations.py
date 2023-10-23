import functions

def mean_squared_error_gd(y, tx, w, max_iters, gamma):
    return functions.mean_squared_error_gd

def mean_squared_error_sgd(y, tx, w, max_iters, gamma,  batch_size=128, num_batches=16): 
    return functions.mean_squared_error_sgd

def least_squares(y, tx): 
    return functions.least_squares

def ridge_regression(y, tx, lambda_):
    return functions.ridge_regression

def logistic_regression(y, tx, w, max_iters, gamma):
    return functions.logistic_regression

def reg_logistic_regression(y, tx, w, max_iters, gamma, lambda_): 
    return functions.reg_logistic_regression