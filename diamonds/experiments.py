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
from pandas.api.types import is_numeric_dtype


BASE_DIR = '../data'
RANDOM_STATE = 42


def remove_zeros(df):
    df = df.drop(df.loc[df.x <= 0].index)
    df = df.drop(df.loc[df.y <= 0].index)
    df = df.drop(df.loc[df.z <= 0].index)
    df = df.drop(df.loc[df.carat <= 0].index)
    df = df.drop(df.loc[df.depth <= 0].index)
    df = df.drop(df.loc[df.table <= 0].index)
    return df


def generate_quantiles(df, low=.05, high=.95):
    quant_df = df.quantile([low, high])
    return quant_df

# Ref: https://gist.github.com/ariffyasri/70f1e9139da770cb8514998124560281


def quantile_removal(df, test, low=.01, high=.99):
    # Removing outliers with the training quantile
    quant_df = generate_quantiles(df, low, high)
    for name in list(df.columns):
        if is_numeric_dtype(df[name]) and name != 'y':
            df = df[(df[name] > quant_df.loc[low, name]) &
                    (df[name] < quant_df.loc[high, name])]
    return df, test


def outliers_removal(train, test):
    train = remove_zeros(train)
    if test:
        test = remove_zeros(test)

    train, _ = quantile_removal(train, test)
    return train, test


def exp_(values):
    return 1 / (1 + np.exp(-values))


def get_sklearn_sgd(params):
    regr = linear_model.SGDRegressor(**params, penalty=None, verbose=True)
    return regr


def load_train_data():
    X = pd.read_pickle("%s/train.pkl" % (BASE_DIR))
    X, _ = outliers_removal(X, None)
    return X


def load_test_data():
    return pd.read_pickle("%s/test.pkl" % (BASE_DIR))


def gen_splits(X, scale=True, exclude_features=None, k=5, test_size=.1):
    X, y = separate_X_y(X, exclude_features)

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


def separate_X_y(X, exclude_features):
    X = X.copy()

    y = X.pop('price')

    if exclude_features:
        X = X.drop(exclude_features, axis=1)

    X = X.values
    y = y.values
    return X, y


def kfold_evaluate(regr, folds, scoring, log_y=False, k=5):
    rmse = []
    mse = []
    mae = []
    r2 = []
    i = 0
    for fold in folds:

        print("Evaluating %s" % (i))
        (X_train, X_val, y_train, y_val) = fold
        if regr == "customSGD":
            if log_y:
                theta = customSGD.SGD(lr=0.1, max_iter=20000,
                X=X_train, y=np.log(y_train), lr_optimizer='invscaling',
                print_interval=2000)
                y_pred = np.exp(customSGD.predict(theta, X_val))
            else:
                theta = normal_equation.normal_equation(X_train, y_train)
                y_pred = customSGD.predict(theta, X_val)
        elif regr:  # Any other Regressor from the SkLearn Library
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
        i += 1

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

    plt.plot([y.min(), y.max()], [y_pred.min(), y_pred.max()], 'k--', lw=3)
    plt.scatter(y, y_pred)
    plt.ylabel('Predicted')
    plt.xlabel('Real')
    plt.show()

    plt.hist(y, bins=100, color='blue', linewidth=3)
    plt.xlabel('Real')
    plt.show()
    plt.hist(y_pred, bins=100, color='red', linewidth=3)
    plt.xlabel('Predicted')
    plt.show()


def fit_evaluate(regr, X_train, X_val, y_train, y_val, log_y=False, scale=False, exclude_features=None):
    print("Evaluating ...")
    if y_val is None:
        X_train, y_train = separate_X_y(X_train, exclude_features)
        X_val, y_val = separate_X_y(X_val, exclude_features)

    if scale:
        scaler = RobustScaler()
        scaler.fit(X_train)
        X_train = scaler.transform(X_train)
        # Fit on train, transforming the test, avoid data leak
        X_val = scaler.transform(X_val)

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

    evaluate(y_val, y_pred)


def fit_evaluate_customSGD(train, test, params={}, log_y=False, scale=False, exclude_features=None):
    print("Evaluating ...")

    X_train, y_train = separate_X_y(train, exclude_features)
    X_test, y_test = separate_X_y(test, exclude_features)

    if scale:
        scaler = RobustScaler()
        scaler.fit(X_train)
        X_train = scaler.transform(X_train)
        # Fit on train, transforming the test, avoid data leak
        X_test = scaler.transform(X_test)

    
    if log_y:
        theta = customSGD.SGD(**params,X=X_train, y=np.log(y_train))
        y_pred = np.exp(customSGD.predict(theta, X_test))
    else:
        theta = customSGD.SGD(**params,X=X_train, y=y_train)
        y_pred = customSGD.predict(theta, X_test)

    evaluate(y_test, y_pred)    
