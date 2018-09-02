import numpy as np
from sklearn import model_selection, ensemble, metrics, linear_model
import matplotlib.pyplot as plt
import pandas as pd
import math
from diamonds import customSGD, normal_equation
from sklearn.preprocessing import RobustScaler
from sklearn.model_selection import KFold
import os
import scipy

BASE_DIR = '../data'
RANDOM_STATE = 42


def exp_(values):
    return 1 / (1 + np.exp(-values))


def get_sklearn_sgd(params):
    regr = linear_model.SGDRegressor(**params, penalty=None, verbose=True)
    return regr


def load_train_data():
    return pd.read_pickle("%s/train.pkl" % (BASE_DIR))


def load_train_test():
    return pd.read_pickle("%s/test.pkl" % (BASE_DIR))


def gen_splits(X, scale=True, exclude_features=None, k=5, test_size=.1):
    X = X.copy()

    y = X.pop('price')

    if exclude_features:
        X = X.drop(exclude_features, axis=1)

    X = X.values
    y = y.values
    if test_size:
        X_train, X_test, y_train, y_test = model_selection.train_test_split(
            X, y, test_size=test_size, random_state=RANDOM_STATE)

    kf = KFold(n_splits=k, random_state=RANDOM_STATE)
    folds = []
    k_idx = 0
    for train_index, val_index in kf.split(X_train):
        k_idx += 1
        X_train_cv, X_val = X_train[train_index].copy(
        ), X_train[val_index].copy()
        y_train_cv, y_val = y_train[train_index].copy(
        ), y_train[val_index].copy()
        if scale:
            scaler = RobustScaler()
            scaler.fit(X_train_cv)
            X_train_cv = scaler.transform(X_train_cv)
            # Fit on train, transforming the validation, avoid data leak
            X_val = scaler.transform(X_val)
        folds.append((X_train_cv, X_val, y_train_cv, y_val))

    # The Scaler must be executed for the full train only after the folds are computed
    # Avoiding data leaks to the Cross Validation
    if scale:
        scaler = RobustScaler()
        scaler.fit(X_train)
        X_train = scaler.transform(X_train)
        # Fit on train, transforming the test, avoid data leak
        X_test = scaler.transform(X_test)

    return folds, (X_train, X_test, y_train, y_test)


def kfold_evaluate(regr, folds, scoring, log_y=False, k=5):
    rmse = []
    mse = []
    mae = []
    r2 = []
    for fold in folds:
        print("Evaluating %s" % (k))
        (X_train, X_val, y_train, y_val) = fold
        if regr:
            regr.verbose = False
            if log_y:
                regr.fit(X_train, np.log(y_train))
                y_pred = np.exp(
                    np.array(regr.predict(X_val), dtype=np.float128))
            else:
                regr.fit(X_train, y_train)
                y_pred = regr.predict(X_val)

        else:
            if log_y:
                theta = normal_equation.normal_equation(
                    X_train, np.log(y_train))
                y_pred = np.exp(customSGD.predict(theta, X_val))
            else:
                theta = normal_equation.normal_equation(X_train, y_train)
                y_pred = customSGD.predict(theta, X_val)

        rmse.append(math.sqrt(((y_pred-y_val)**2).mean()))
        mse.append(metrics.mean_squared_error(y_val, y_pred))
        mae.append(metrics.mean_absolute_error(y_val, y_pred))
        r2.append(metrics.r2_score(y_val, y_pred))

    print("RMSE: \t %.4f +/- %.4f" % (np.mean(rmse), np.std(rmse)))
    print("MSE:  \t %.4f +/- %.4f" % (np.mean(mse), np.std(mse)))
    print("MAE:  \t %.4f +/- %.4f" % (np.mean(mae), np.std(mae)))
    print('R2:   \t %.4f +/- %.4f' % (np.mean(r2), np.std(r2)))


def evaluate(y, y_pred):
    error = math.sqrt(((y_pred-y)**2).mean())
    print("RMSE : %.4f" % error)
    print("MSE: %.4f" % metrics.mean_squared_error(y, y_pred))
    print("MAE: %.4f" % metrics.mean_absolute_error(y, y_pred))
    print('R2: %.4f' % metrics.r2_score(y, y_pred))

    plt.hist(y, bins=100, color='blue', linewidth=3)
    plt.show()
    plt.hist(y_pred, bins=100, color='red', linewidth=3)
    plt.show()
