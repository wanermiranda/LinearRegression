# LinearRegression

Repository for the first Machine Learning project related to Linear Regression, as part of the Machine Learning subject at IC Unicamp

## Data Exploratory Analysis

- Feature engineering
- Feature Scaling — Fixed ## (Done)
  - Robust Scaler
  - Gaussian Scale
  - Quantiles (0.25, 0.75)
  - New Features ## (Done)
  - RatioXY
  - RatioXZ
  - Volume
- Removing outliers --- ## (Done)
  - Removing samples with any features equal to 0
- Run Scenario Exps K Fold CV with a fixed parameters ## Azael
  - Crude ## (Done) - Did not converge due to high avg. loss
  - Scaled ## (Done) - Great results (R2: 0.904)
  - Add Sintect Features and run Sklearn Feature Selector

## Experimental Evaluation

- Metrics Definition
  - MSE (mean square error)
  - RMSE (root-mean-square error)
  - MAE (mean absolute error)
  - R2 (variance)
- Silhouette visualization
  - Y values Histogram
  - Y and Y’ values comparison with a trend line
  - Loss visualization
  - MSE overtime
- Implement Regularization and Learning Rates Reg
  - Inv Scaling
  - Step function
  - Regularization Factor
- Implement different approaches for Gradient Descent
  - Mini Batch.
  - Stochastic Batch.
  - Batch
- Implement Normal Equation

## Experimental Execution Report (Sklearn vs Custom)

Rationale: Since the SGD from SKlearn is already tested by the community we will use its executions as a baseline to compare its performance with our custom implementation.

- (Sklearn) GridSearch for HyperParameters by K-Fold Cross Validation
  - Learning rate Values
  - Learning rate Regularization
  - Number Iterations
- (Custom) Apply the parameters found on item (b), and compare the results.

- Consolidate the results from the 3 steps above into the final report.

```python
# Epoch: When the training data was used.
# Iteration: The batch updates
from sklearn.utils import shuffle

def getBatch(X, y, it, b_it, b_sz, epoch):
    b_ct = int(X.shape[0]/sz)
    b_it += 1
    If b_it > b_ct:
        b_it = 1
    X_ = X[b_it * b_sz: (b_it+1) * b_sz]
    return X_, y_, b_it, epoch


def SGD_(alpha, max_iter, X, y):

#Creating theta0
    X = np.insert(X, values=1, obj=0, axis=1)
    shape = X.shape
    nsamples = shape[0]
    print("Number of samples: "+str(nsamples))
    theta0 = np.zeros(nsamples)
    nparams = shape[1]
    print("Number of parameters: "+str(nparams))
    theta = np.random.uniform(size=nparams)
    theta_temp = np.ones(nparams)
    error = 1
    epsilon = 0.001
    it = 0
    i = 0
    power_t = 0.25
    t=1.0
    while ((error > epsilon) and (it < max_iter)):
	If epoch changes and its stochastic: :
		X, y = shuffle(X, y, random_state=0)
       X_, y_ = getBatch(X, y, it, sz)
        h0 = hyphotesis(theta, X)
        eta = alpha / pow(t, power_t)
        error = (h0 - y)
        for j in range(nparams):
            theta_temp[j] = MSE_theta(theta, X, y, eta, j, h0, error)
        it += 1
        i += 1
        y_pred = hyphotesis(theta_temp, X)
        error = ((y - y_pred) ** 2).mean() / 2
        theta = theta_temp.copy()
        if (i % 100) == 0 or i == 1:
            print("Epoch: %s Batch: %s Error: %.8f lr: %.8f "%(it, i, error, eta))
        t += 1

    return theta
```
