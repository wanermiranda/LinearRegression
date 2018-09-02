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
    return theta


def normal_equation_test():
	X_, y_ = customSGD.get_toy_data_big()
	X,  X_val, y, y_val = model_selection.train_test_split(X_, y_, test_size=0.2, random_state=42)
	theta = normal_equation(X, y)
	y_pred = customSGD.predict(theta, X_val)
	error = math.sqrt(((y_pred-y_val)**2).mean())
	print("RMSE error: %.4f" % error)
	print("MSE: %.3f" % metrics.mean_squared_error(y_val, y_pred))
	print("MAE: %.3f" % metrics.mean_absolute_error(y_val, y_pred))
	print('R2: %.3f' % metrics.r2_score(y_val, y_pred))
