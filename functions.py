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

############################# Step 2 #############################
def compute_loss_mse(y, tx, w):
    """Calculate the loss using mse."""
    e = y - tx.dot(w)
    loss = e.dot(e) / (2 * len(e))
    return loss

def compute_gradient_mse(y, tx, w):
    """Compute the gradient."""
    e = y - tx.dot(w)
    gradient = - tx.T.dot(e) / len(e)
    return gradient

def compute_loss_logistic(y, tx, w):
    """compute the cost by negative log likelihood."""
    pred = tx.dot(w)
    loss = np.sum(np.log(1 + np.exp(pred)) - y * pred)
    return loss

def compute_gradient_logistic(y, tx, w):
    """compute the gradient of loss."""
    pred = tx.dot(w)
    gradient = tx.T.dot(sigmoid(pred) - y)
    return gradient


def mean_squared_error_gd(y, tx, initial_w, max_iters, gamma):
    """calculate the loss by mse."""
    w = initial_w
    for n_iter in range(max_iters):
        gradient = compute_gradient_mse(y, tx, w)
        loss = compute_loss_mse(y, tx, w)
        w = w - gamma * gradient
    return w, loss


def mean_squared_error_sgd(y, tx, initial_w, max_iters, gamma):
    """calculate the loss by mse."""
    w = initial_w
    for n_iter in range(max_iters):
        for minibatch_y, minibatch_tx in batch_iter(y, tx, batch_size=1, num_batches=1):
            gradient = compute_gradient_mse(minibatch_y, minibatch_tx, w)
            loss = compute_loss_mse(minibatch_y, minibatch_tx, w)
            w = w - gamma * gradient
    return w, loss


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
    for n_iter in range(max_iters):
        gradient = compute_gradient_logistic(y, tx, w)
        loss = compute_loss_logistic(y, tx, w)
        w = w - gamma * gradient
    return w, loss


def reg_logistic_regression(y, tx, lambda_, initial_w, max_iters, gamma):
    """implement regularized logistic regression."""
    w = initial_w
    for n_iter in range(max_iters):
        gradient = compute_gradient_logistic(y, tx, w) + 2 * lambda_ * w
        loss = compute_loss_logistic(y, tx, w) + lambda_ * np.squeeze(w.T.dot(w))
        w = w - gamma * gradient
    return w, loss







