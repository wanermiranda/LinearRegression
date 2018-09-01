import sys
sys.path.append('../')
import numpy as np  # linear algebra
from numpy.core.umath_tests import inner1d
import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)
from pandas.api.types import CategoricalDtype
from sklearn import model_selection, ensemble, metrics, linear_model
import matplotlib.pyplot as plt
import os
import math
from diamonds import customSGD


#  For linear regression, it is possible to estimate the values
# of all parameters theta by applying the normal equation method,
# which corresponds to the following equation:
#
#  Theta = (Xt.X)^-1.Xt.y
#
#  This procedure is called Normal Equation, which is implemented
# here
#
# params:
#   X -> set of features
#   Y -> set of targets
#
# return:
#   theta -> set of parameters
#
def normal_equation(X, y):
    X = np.insert(X, 0, 1, axis=1)
    npX = np.copy(X)
    npY = y.transpose()
    npXt = npX.transpose()

    R1 = np.matmul(npXt, npX)

    det = np.linalg.det(R1)

    if (det != 0):
        R1 = np.linalg.inv(R1)
        R2 = np.matmul(npXt, npY)
        theta = np.matmul(R1, R2)
    else:
        theta = []
        print("Error! Matrix (Xt.X) has no inverse.")
    theta = np.array(theta)
    print('Theta', theta)
    return theta


def normal_equation_test():
    X, y = customSGD.get_toy_data()
    theta = normal_equation(X, y)
    y_hat = customSGD.predict(theta, X)
    error = math.sqrt(((y_hat-y)**2).mean())
    print("RMSE error: %.4f" % error)
