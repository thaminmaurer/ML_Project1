import numpy as np
import matplotlib.pyplot as plt


### standardize
def standardize(x):
    """Standardize the original data set."""
    mean_x = np.mean(x, axis=0)
    x = x - mean_x
    std_x = np.std(x, axis=0)
    x = x / (std_x+1e-12) ### the small term is to deal with the division by zero 
    return x, mean_x, std_x


def compute_sigmoid(x):
    """ 
    Sigmoid function for logistic regression
    Args:
    x: input data
    """
    
    sigmoid = 1/(1+np.exp(-x))
    return sigmoid


def handle_nan_values(x, delete_nan_columns=False):
    # Calculate dimension of x
    D = x.shape[1]

    #calculate number number of nan per column
    logical_matrix = np.isnan(x)
    nan_per_columns = np.sum(logical_matrix, axis=0)
    average_nan = np.mean(nan_per_columns)

    # delet the columns with more nan than the average
    if delete_nan_columns:
        x = x[:, nan_per_columns <= average_nan]

    # Replace NaN entries with mean
    for i in range(D):
        nan_entries = np.isnan(x[:,i])
        mean = np.mean(x[~nan_entries,i])
        x[nan_entries, i] = mean
    return x


def predict_y(w, x):
    y_pred=x.dot(w)
    y_pred = compute_sigmoid(y_pred) # now y_pred values should be between 0 and 1
    y_pred[y_pred>0.5] = 1
    y_pred[y_pred<=0.5] = -1
    return y_pred
 

############################# Step 2 #############################
def compute_loss_mse(y, tx, w):
    """Calculate the loss using mse."""
  
    e = y - tx.dot(w)
    loss = e.T.dot(e) / (2 * len(e))
    return loss

def compute_gradient_mse(y, tx, w):
    """Compute the gradient."""
    e = y - tx.dot(w)
    gradient = - tx.T.dot(e) / len(e)
    return gradient

def compute_loss_logistic(y, tx, w):
    """compute the cost by negative log likelihood."""
    pred = tx.dot(w)
    loss = np.sum(np.log(1 + np.exp(pred)) - y * pred) / len(pred)
    return loss

def compute_loss_logistic_minusone_one(y, tx, w):
    """compute the cost by negative log likelihood."""
    pred = tx.dot(w)
    loss = np.sum(np.log(1 + np.exp(-y*pred))) / len(pred)
    return loss

def compute_gradient_logistic(y, tx, w):
    """compute the gradient of loss."""
    pred = tx.dot(w)
    gradient = (tx.T.dot(compute_sigmoid(pred) - y))/ len(y)
    return gradient

def compute_gradient_logistic_minusone_one(y, tx, w):
    """compute the gradient of loss."""
    pred = tx.dot(w)
    gradient = - (tx.T.dot(y * (1 - (compute_sigmoid(y*pred) )))) / len(y)
    return gradient

def batch_iter(y, tx, batch_size, num_batches=1, shuffle=True):
    """
    Generate a minibatch iterator for a dataset.
    Takes as input two iterables (here the output desired values 'y' and the input data 'tx')
    Outputs an iterator which gives mini-batches of `batch_size` matching elements from `y` and `tx`.
    Data can be randomly shuffled to avoid ordering in the original data messing with the randomness of the minibatches.
    Example of use :
    for minibatch_y, minibatch_tx in batch_iter(y, tx, 32):
        <DO-SOMETHING>
    """
    
    data_size = len(y)
    
    if shuffle:
        shuffle_indices = np.random.permutation(np.arange(data_size))
        shuffled_y = y[shuffle_indices]
        shuffled_tx = tx[shuffle_indices]
        
    else:
        shuffled_y = y
        shuffled_tx = tx
        
    for batch_num in range(num_batches):
        start_index = batch_num * batch_size
        end_index = min((batch_num + 1) * batch_size, data_size)
        
        yield shuffled_y[start_index:end_index], shuffled_tx[start_index:end_index]


def mean_squared_error_gd(y, tx, initial_w, max_iters, gamma):
    """calculate the loss by mse."""
    w = initial_w
    loss = compute_loss_mse(y, tx, w)
    for n_iter in range(max_iters):
        gradient = compute_gradient_mse(y, tx, w)
        w = w - gamma * gradient
        loss = compute_loss_mse(y, tx, w)
    return w, loss


def mean_squared_error_sgd(y, tx, initial_w, max_iters, gamma,  batch_size=1, num_batches=1):
    """calculate the loss by mse."""
    w = initial_w
    best_w = []
    best_loss = 1e16
    for n_iter in range(max_iters):
        for minibatch_y, minibatch_tx in batch_iter(y, tx, batch_size, num_batches):
            gradient = compute_gradient_mse(minibatch_y, minibatch_tx, w)
            w = w - gamma * gradient
            loss = compute_loss_mse(minibatch_y, minibatch_tx, w)
            if loss < best_loss:
                best_loss = loss
                best_w = w
    return best_w, best_loss


def least_squares(y, tx):
    """calculate the least squares solution."""
    a = tx.T.dot(tx)
    b = tx.T.dot(y)
    w = np.linalg.solve(a, b)
    loss = compute_loss_mse(y, tx, w)
    return w, loss


def ridge_regression(y, tx, lambda_):
    """implement ridge regression."""
    aI = 2 * tx.shape[0] * lambda_ * np.identity(tx.shape[1])
    a = tx.T.dot(tx) + aI
    b = tx.T.dot(y)
    w = np.linalg.solve(a, b)
    loss = compute_loss_mse(y, tx, w)
    return w, loss


def logistic_regression(y, tx, initial_w, max_iters, gamma):
    """implement logistic regression."""
    w = initial_w
    loss = compute_loss_logistic_minusone_one(y, tx, w)
    for n_iter in range(max_iters):
        gradient = compute_gradient_logistic_minusone_one(y, tx, w)
        w = w - gamma * gradient
        loss = compute_loss_logistic_minusone_one(y, tx, w)
    return w, loss


def reg_logistic_regression(y, tx, initial_w, max_iters, gamma, lambda_):
    """implement regularized logistic regression."""
    w = initial_w
    for n_iter in range(max_iters):
        gradient = compute_gradient_logistic(y, tx, w) + 2 * lambda_ * w
        loss = compute_loss_logistic(y, tx, w) + lambda_ * np.squeeze(w.T.dot(w))
        w = w - gamma * gradient
    return w, loss


def logistic_regression_with_mb(y, tx, initial_w, max_iters, gamma, batch_size):
    w = initial_w
    num_samples = len(y)
    
    for n_iter in range(max_iters):
        for batch_start in range(0, num_samples, batch_size):
            batch_end = batch_start + batch_size
            batch_tx = tx[batch_start:batch_end]
            batch_y = y[batch_start:batch_end]
            
            gradient = compute_gradient_logistic_minusone_one(batch_y, batch_tx, w)
            loss = compute_loss_logistic_minusone_one(batch_y, batch_tx, w)
            
            w = w - gamma * gradient
        
    return w, loss








