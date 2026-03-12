# This script is dedicated to the flow-based optimal transport 
# barycenter solvers. 

import numpy as np

def gaussian_kernel(X, sigma=1.0): 
    """
    Computes the Gaussian kernel matrix for the given data points. 

    Inputs: 
        - x: the data points. N-by-d numpy array where N is the number of 
            samples and d is the dimension of each x_i vector. 
        - sigma: the standard deviation of the Gaussian kernel. 
    """
    # Compute the squared Euclidean distance matrix
    distances = np.sum(X**2, axis=1, keepdims=True) - 2 * X @ X.T + np.sum(X.T**2, axis=0, keepdims=True)
    # print("distances: {}".format(distances))
    # Compute the kernel matrix using the Gaussian kernel
    K_x = np.exp(-distances / (2 * sigma**2))
    return K_x


def gaussian_kernel_grad(y, i, l, gauss_kernel, sigma=1.0, verbose=0, second_kernel = None): 
    """ 
    Returns the gradient of the kernel matrix at entry (i, l) with respect the index of y. 
    """
    if second_kernel is None: 
        result = np.multiply(-gauss_kernel[i, l].reshape((y.shape[0], 1)), y[i, :] - y[l, :]) / sigma**2
        if verbose > 10: print(f"Gaussian Kernel Grad (i = {i}, l = {l}); Use Case 1: \n{result}")
    else: 
        result = np.multiply(-(gauss_kernel[i, l] * second_kernel[i, l]).reshape((y.shape[0], 1)), y[i, :] - y[l, :]) / sigma**2
        if verbose > 10: print(f"Gaussian Kernel Grad (i = {i}, l = {l}); Use Case 2: \n{result}")
    if verbose > 10: 
        print("Shape of gradient = {}".format(result.shape))
    return result

def gaussian_kernel_kl_grad(y, x, lam, k_y, k_z, verbose = 0): 
    """
    Computes the gradient of the loss function with respect to y. 
    """
    # compute the gradient of the kernel matrix
    grad = np.zeros_like(y)
    for i in range(y.shape[0]): 
        if verbose > 2:
            print("Iteration {}".format(i))
        grad[i] = np.sum(gaussian_kernel_grad(y, i, np.arange(y.shape[0]), k_y, verbose = verbose, second_kernel = k_z), axis=0) / np.sum(k_z[i, :] * k_y[i, :])
        if i == 0 and verbose > 10:     
            print(f"Numerator Sum: \n{np.sum(gaussian_kernel_grad(y, i, np.arange(y.shape[0]), k_y, verbose=verbose, second_kernel = k_z), axis=0)} \nwith dimensions {np.sum(gaussian_kernel_grad(y, i, np.arange(y.shape[0]), k_y, verbose=verbose, second_kernel = k_z), axis=0).shape}")
            print("Gradient after update one: \n{}\n".format(grad[i]))
        grad[i] -= np.sum(gaussian_kernel_grad(y, i, np.arange(y.shape[0]), k_y, verbose = verbose), axis=0) / np.sum(k_y[i, :])
        if i == 0 and verbose > 10: 
            print(f"Numerator Sum 2: \n{np.sum(gaussian_kernel_grad(y, i, np.arange(y.shape[0]), k_y, verbose=verbose), axis=0)} \nwith shape {np.sum(gaussian_kernel_grad(y, i, np.arange(y.shape[0]), k_y, verbose=verbose), axis=0).shape}")
            print("Gradient after update two: \n{}".format(grad[i]))
        # print("Dimension of Gradient: {}".format(grad[i].shape))
        # print("Dimension of y {}".format(y.shape))
        # print("Dimension of x {}".format(x.shape))
        # print("Dimension of lam {}".format(lam.shape))
        grad[i] = y[i] - x[i] + lam * grad[i]
    return grad

def compute_barycenter(x, z, y_init, lam, barycenter_cost_grad=gaussian_kernel_kl_grad, kern_y=gaussian_kernel, kern_z=gaussian_kernel, 
                       epsilon=0.001, lr=0.01, max_iter=1000, verbose=0, adaptive_lr=False, growing_lambda=True, 
                       warm_stop = 200, max_lambda = 300, monitor=None): 
    """
    Computes the barycenter with a flow-based approach. In other words, 
    we run gradient descent on y_i with respect to the barycenter objective
    until it reaches convergence. 

    Inputs: 
        - x: the observed data points. N-by-d numpy array where N is the number 
            of samples and d is the dimension of each x_i vector. 
        - z: the hidden factors. N-by-k numpy array where k is the dimension of 
            each z_i vector. 
        - y_init: the initial starting points of the barycenter points. N-by-d 
            numpy array.
        - lam: the regularization parameter for controlling the independence 
            between y and z.
        - barycenter_cost_grad: a function that computes the gradient of the 
            barycenter objective with respect to y. 
        - kern_y: the kernel to use for estimating the distribution of y
        - kern_z: the kernel to use for estimating the distribution of z
        - epsilon: the convergence threshold.
        - lr: the learning rate for gradient descent.
        - max_iter: the maximum number of iterations to run.
        - verbose: the verbosity level for debugging 
        - adaptive_lr: whether to use adaptive learning rate or not.
    """
    y = y_init
    iter = 0
    # pre-compute the kernel matrix for Z since that remains constant
    k_z = kern_z(z)
    old_grad_norm = float('inf')
    # pre-compute the growth rate of the lambda and the stopping iteration
    if growing_lambda: 
        lambda_growth = (max_lambda - lam) / warm_stop
    if monitor is not None: 
        monitor_iters = 0
        monitor.eval({"y": y, "Lambda": lam, "Iteration": iter, "Gradient Norm": None})
    # iterate until maximum iteration or convergence
    while iter < max_iter: 
        # update the iteration count 
        iter += 1
        k_y = kern_y(y)
        if iter == 1 and verbose == True: 
            print("Kernel Y from First Iteration:\n", k_y)
        # compute the gradient vector of this iteration
        grad = barycenter_cost_grad(y, x, lam, k_y, k_z, verbose=verbose)
        if iter == 1: 
            print("Gradient from First Iteration:\n", grad)
        # run a gradient descent step
        y = y - lr * grad
        # check for convergence
        if iter > warm_stop and np.linalg.norm(grad) < epsilon:
            break
        # adaptive update of the learning rate
        if adaptive_lr and lr < 1 and lr > 0.0001: 
            lr = lr * 1.01 if (np.linalg.norm(grad) < old_grad_norm) else lr * 0.5
        old_grad_norm = np.linalg.norm(grad)
        # print the gradient norm every 100 iterations
        if verbose >= 1 and iter % 100 == 0:
            print("Iteration {}: gradient norm = {}".format(iter, np.linalg.norm(grad)))
        # update the lambda value if necessary
        if growing_lambda and iter < warm_stop: 
            lam += lambda_growth
        # perform monitor functionality if necessary 
        if monitor is not None and monitor_iters == 0: 
            monitor.eval({"y": y, "Lambda": lam, "Iteration": iter, "Gradient Norm": old_grad_norm})
            monitor_iters = monitor.get_monitoring_skip()
        if monitor is not None and monitor_iters != 0: 
            monitor_iters -= 1
    # print the final gradient norm and number of iterations
    if verbose >= 2 :
        print("Final gradient norm = {}".format(np.linalg.norm(grad)))
        print("Number of iterations = {}".format(iter))
    return y

def gaussian_kernel_single(x_i, x_l, sigma): 
    """
    Computes a single entry of the gaussian kernel
    """
    return np.exp(np.linalg.norm(x_i - x_l) / (-2 * sigma ** 2))

def gaussian_kernel_duo(Y, Y_center, sigma = 1.0): 
    """
    Compute the kernel matrix using Y and Y_center
    """
    n = Y.shape[0]
    kern = np.zeros((n, n))
    for i in range(n): 
        for l in range(n): 
            kern[i, l] = np.exp(-(np.linalg.norm(Y[i, :] - Y_center[l, :])) / (2 * sigma**2))
    return kern

def kl_barycenter_loss(Y, Y_center, X, Z, lam, kern_y = gaussian_kernel_duo, kern_z = gaussian_kernel, verbose = 0): 
    """
    Computes the loss function of the barycenter problem.

    Inputs: 
        - X: the observed data points. N-by-d numpy array where N is the number
            of samples and d is the dimension of each x_i vector.
        - Y: the barycenter points. N-by-d numpy array.
        - Z: the hidden factors. N-by-k numpy array where k is the dimension of
            each z_i vector.
        - lam: the regularization parameter for controlling the independence
            between y and z.
        - kern_y: the kernel to use for estimating the distribution of y
        - kern_z: the kernel to use for estimating the distribution of z 
    """
    # compute the kernel matrices
    k_y = kern_y(Y, Y_center)
    k_z = kern_z(Z)
    # compute the loss function
    loss = 0
    for i in range(X.shape[0]): 
        temp = 0
        temp += np.linalg.norm(Y[i, :] - X[i, :])**2 / 2
        temp += lam * np.log(np.sum(k_y[i, :] * k_z[i, :]) / (np.sum(k_y[i, :]) * np.sum(k_z[i, :])))
        if verbose > 0: 
            print(f"The loss value evaluation for entry i = {i}: {temp}")
        loss += temp 
    return loss


        

                
