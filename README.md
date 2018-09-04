# LinearRegression

Repository for the first Machine Learning project related to Linear Regression, as part of the Machine Learning subject at IC Unicamp

## Structure

The project was organized in 3:

* `./data` — A directory containing the `diamonds.csv` downloaded from kaggle and output files for train and test sets.
* `./notebooks` — The directory containing the notebooks used to execute each step of the experiments.
* `./diamonds` — A python module directory for the raw implementations for experiments, normal equation and customSGD.

## Requirements

`$ pip install -r requirements.txt --user`

* backcall==0.1.0
* cycler==0.10.0
* decorator==4.3.0
* ipykernel==4.9.0
* ipython==6.5.0
* ipython-genutils==0.2.0
* jedi==0.12.1
* jupyter-client==5.2.3
* jupyter-core==4.4.0
* kiwisolver==1.0.1
* matplotlib==2.2.3
* numpy==1.15.1
* pandas==0.23.4
* parso==0.3.1
* pexpect==4.6.0
* pickleshare==0.7.4
* prompt-toolkit==1.0.15
* ptyprocess==0.6.0
* pycurl==7.43.0
* Pygments==2.2.0
* pygobject==3.20.0
* pyparsing==2.2.0
* python-apt==1.1.0b1
* python-dateutil==2.7.3
* pytz==2018.5
* pyzmq==17.1.2
* scikit-learn==0.19.2
* scipy==1.1.0
* simplegeneric==0.8.1
* six==1.11.0
* sklearn==0.0
* tornado==5.1
* traitlets==4.3.2
* wcwidth==0.1.7

## Reproduction

The reproduction of work should be done by executing the notebooks in the following order:
 
 * `Dataset Preparation.ipynb` First notebook to be executed, it builds the train and test datasets while creating syntetic features and cleaning zero-ed values.
 * `Custom SGD Tests.ipynb` This notebook is used to test the implementations of the custom methods, SGD and Normal equation.
 * `Experiments - Round 1.ipynb` Using the normal equation this notebook runs various 5k fold experiments to compare the feature engineering contributions.
 * `Experiments - Round 2.ipynb` In this we evalute the GridSearch for hyperparameters using the Sklearn SGDRegressor as a base implementation.
 * `Experiments - Round 3.ipynb` The round 3 is reponsible for the comparison between the three methods: Normal Equation, SGDRegressor and the customSGD
 * `Experiments - Round 4.ipynb` This round was to identify is there is any overfitting when running 1 million iterations.
 * `Experiments - Round 5.ipynb` The last experiment using the whole training set and evaluating the predictions on the reserved test set.

## Project Planning 

The project was done follwwing the bellow guidelines in order to be finished.

### Data Exploratory Analysis

- Feature engineering ( Done in the Dataset preparations notebook )
- Feature Scaling — Fixed (Done)
  - Robust Scaler — (Done)
  - Gaussian Scale — (Done)
  - Quantiles (0.25, 0.75) — (0.01, 0.99) (Done)
  - New Features (Done)
  - RatioXY (Done)
  - RatioXZ (Done)
  - Volume (Done)
- Removing outliers --- (Done)
  - Removing samples with any features equal to 0 (Done)
- Run Scenario Exps K Fold CV with a fixed parameters (Azael)
  - Crude (Done) — Did not converge due to high avg. loss (Done fixed and moved to a module called experiments)
  - Scaled (Done) — Great results (R2: 0.904)
  - Create a notebook to help the evaluation while creating also helpers (Done)
  - Add Sintect Features and run Sklearn Feature Selector (Done by manual selecion using the Normal Equation)

### Experimental Evaluation

- Metrics Definition (Done Included in the method evaluate in the experiments module)
  - MSE (mean square error) (Done)
  - RMSE (root-mean-square error) (Done)
  - MAE (mean absolute error) (Done)
  - R2 (variance) (Done)
- Silhouette visualization - (Done Included in the method evaluate in the experiments module)
  - Y values Histogram (Done)
  - Y and Y’ values comparison with a trend line (Done)
  - Loss visualization
  - MSE overtime
- Implement Regularization and Learning Rates Reg
  - Inv Scaling (Done)
  - Step function
  - Regularization Factor
- Implement different approaches for Gradient Descent (Done - Custom SGD Tests notebook)
  - Mini Batch. (Done)
  - Stochastic Batch. (Done)
  - Batch (Done)
- Implement Normal Equation (Done by Azael - Included in the normal_equation module)

### Experimental Execution Report (Sklearn vs Custom)

Rationale: Since the SGD from SKlearn is already tested by the community we will use its executions as a baseline to compare its performance with our custom implementation.

- (Sklearn) GridSearch for HyperParameters by K-Fold Cross Validation
  - Learning rate Values
  - Learning rate Regularization
  - Number Iterations
- (Custom) Apply the parameters found on item (b), and compare the results.

- Consolidate the results from the 3 steps above into the final report.
