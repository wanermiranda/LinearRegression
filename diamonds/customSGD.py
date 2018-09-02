import numpy as np  # linear algebra
from numpy.core.umath_tests import inner1d
import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)
from pandas.api.types import CategoricalDtype
from sklearn import model_selection, ensemble, metrics, linear_model
import matplotlib.pyplot as plt
import os
import math
from sklearn.utils import shuffle
from sklearn.datasets import make_regression
import timeit
import time 

def get_batch(X, y, b_it, b_sz, epoch):
    b_ct = int(X.shape[0]/b_sz)
    y_ = np.zeros((0, 0))
    X_ = np.zeros((0, 0))
    start = 0
    finish = 0

    if b_it > b_ct:
        b_it = 0
        epoch += 1
    start = b_it * b_sz
    finish = (b_it+1) * b_sz
    X_ = X[start: finish]
    y_ = y[start: finish]

    b_it += 1    

    return X_, y_, b_it, epoch


def get_batch_test():
    count = 105
    X = np.ones((count, 3))
    y = np.array(range(0, count))
    b_it = 0
    b_sz = 100
    epoch = 0
    it = 0
    sz = 0
    while epoch < 1:
        X_, y_, b_it, epoch = get_batch(X, y, b_it, b_sz, epoch)
        sz += X_.shape[0]
        print(it, sz, X_.shape, y_.shape, b_it, b_sz, epoch)
        print(y_)
        it += 1


def hypothesis(theta, X):
    return np.sum(theta.T * X, axis=1)


def grad_mse_step(theta, X, y, alpha, j, h0, error):
    S = np.sum(np.matmul(error, X[:, j]))
    result = theta[j] - (alpha * (1. / len(y)) * S)
    return result


def grad_mse_step_test():
    theta = np.array([1, 0, 0], dtype='float64')
    theta_temp = np.array([0, 0, 0], dtype='float64')
    X, y = get_toy_data()
    X = np.insert(X, 0, 1, axis=1)

    print("X values ")
    print(X)
    alpha = .01
    max_iter = 50
    for i in range(max_iter):
        h0 = hypothesis(theta, X)
        error = (h0 - y)
        for j in range(X.shape[1]):
            theta_temp[j] = grad_mse_step(theta, X, y, alpha, j, h0, error)

        theta = theta_temp.copy()
        print("Iter %i theta: %s" % (i, theta))
        y_hat = hypothesis(theta, X)
        error = math.sqrt(((y_hat-y)**2).mean())
        print("RMSE error: %.4f" % error)

    theta = np.array(theta)
    print("Predicted: %s" % (hypothesis(theta, X)))
    print("Expected: %s" % (y))


def get_toy_data():
    y = np.array([2., 4.], dtype='float64')
    X = np.array([[4., 7.], [2., 6.]], dtype='float64')
    return X, y

def get_toy_data_big():
    return make_regression(n_samples=50000, n_features=25, noise=0.5)


def SGD(lr, max_iter, X, y, lr_optimizer=None,
        epsilon=0.001, power_t=0.25, t=1.0,
        batch_type='Full',
        batch_sz=1,
        print_interval=100):

    # Adding theta0 to the feature vector
    X = np.insert(X, values=1, obj=0, axis=1)

    shape = X.shape
    nsamples = shape[0]
    print("Number of samples: "+str(nsamples))
    nparams = shape[1]
    print("Number of parameters: "+str(nparams))

    theta = np.random.uniform(size=nparams)
    theta_temp = np.ones(nparams)

    error = 1
    it = 0
    epoch = 0
    lst_epoch = 0
    b_it = 0

    if batch_type == 'Full':
        b_sz = nsamples
    else:  # Mini or Stochastic
        b_sz = batch_sz

    if batch_type == 'Stochastic':
        X, y = shuffle(X, y)
        print ('Shuffled')

    while ((error > epsilon) and (it < max_iter)):
        if lr_optimizer == 'invscaling':
            eta = lr / pow(t, power_t)
        else:
            eta = lr

        X_ = np.zeros(0)
        y_ = np.zeros(0)
        while y_.shape[0] == 0:
            # Checking if it is a new epoch to shuffle the data.            
            X_, y_, b_it, epoch = get_batch(X, y, b_it, b_sz, epoch)
            if lst_epoch < epoch:
                lst_epoch = epoch
                if batch_type == 'Stochastic':
                    X, y = shuffle(X, y)

        h0 = hypothesis(theta, X_)

        error = (h0 - y_)
        for j in range(nparams):
            theta_temp[j] = grad_mse_step(theta, X_, y_, eta, j, h0, error)

        y_pred = hypothesis(theta_temp, X_)
        error = ((y_ - y_pred) ** 2).mean() / 2

        theta = theta_temp.copy()

        it += 1
        t += 1

        if (it % print_interval) == 0 or it == 1:
            print("It: %s Batch: %s Epoch %i Error: %.8f lr: %.8f " %
                  (it, b_it, epoch, error, eta))
    print("Finished \n It: %s Batch: %s Epoch %i Error: %.8f lr: %.8f " %
                  (it, b_it, epoch, error, eta))
    return theta


def predict(theta, X):
    X = np.insert(X, values=1, obj=0, axis=1)
    return hypothesis(theta, X)


def SGD_test():
    X_, y_ = get_toy_data_big()
    X,  X_val, y, y_val = model_selection.train_test_split(X_, y_, test_size=0.2, random_state=42)
    print("X values ")
    print(X)
    lr = .01
    max_iter = 2000
    batch_sz = 100
    
    print ("")
    print ("Full batch")

    start = time.process_time()
    theta = SGD(lr, max_iter, X, y, batch_type='Full', print_interval=100)
    print("finished ", time.process_time() - start)
    y_pred = predict(theta, X_val)
    print("MSE: %.3f" % metrics.mean_squared_error(y_val, y_pred))
    print("MAE: %.3f" % metrics.mean_absolute_error(y_val, y_pred))
    print('R2: %.3f' % metrics.r2_score(y_val, y_pred))


    print ("")
    print ("Stochastic Mini batch")
    start = time.process_time()
    theta = SGD(lr, max_iter, X, y, batch_type='Stochastic', batch_sz=batch_sz, print_interval=100)
    print("finished ", time.process_time() - start)
    y_pred = predict(theta, X_val)
    print("MSE: %.3f" % metrics.mean_squared_error(y_val, y_pred))
    print("MAE: %.3f" % metrics.mean_absolute_error(y_val, y_pred))
    print('R2: %.3f' % metrics.r2_score(y_val, y_pred))

    print ("")
    print ("Mini batch")
    start = time.process_time()
    theta = SGD(lr, max_iter, X, y, batch_type='Mini',  batch_sz=batch_sz, print_interval=100)
    print("finished ", time.process_time() - start)
    y_pred = predict(theta, X_val)
    print("MSE: %.3f" % metrics.mean_squared_error(y_val, y_pred))
    print("MAE: %.3f" % metrics.mean_absolute_error(y_val, y_pred))
    print('R2: %.3f' % metrics.r2_score(y_val, y_pred))

    print ("")
    print ("Single Instance")
    start = time.process_time()
    theta = SGD(lr, max_iter, X, y, batch_type='Single',  epsilon=10**-10, batch_sz=1, print_interval=100)
    print("finished ", time.process_time() - start)
    y_pred = predict(theta, X_val)
    print(y_pred.shape, y_val.shape)
    print("MSE: %.3f" % metrics.mean_squared_error(y_val, y_pred))
    print("MAE: %.3f" % metrics.mean_absolute_error(y_val, y_pred))
    print('R2: %.3f' % metrics.r2_score(y_val, y_pred))

