# LinearRegression

Repository for the first Machine Learning project related to Linear Regression, as part of the Machine Learning subject at IC Unicamp

## Data Exploratory Analysis

- Feature engineering ( Done in the Dataset preparations notebook )
- Feature Scaling — Fixed (Done)
  - Robust Scaler - (Done)
  - Gaussian Scale - (Done)
  - Quantiles (0.25, 0.75) - (0.01, 0.99) (Done)
  - New Features (Done)
  - RatioXY (Done)
  - RatioXZ (Done)
  - Volume (Done)
- Removing outliers --- (Done)
  - Removing samples with any features equal to 0 (Done)
- Run Scenario Exps K Fold CV with a fixed parameters (Azael)
  - Crude ## (Done) - Did not converge due to high avg. loss (Done fixed and moved to a module called experiments)
  - Scaled ## (Done) - Great results (R2: 0.904)
  - Create a notebook to help the evaluation while creating also helpers (Done)
  - Add Sintect Features and run Sklearn Feature Selector (Done by manual selecion using the Normal Equation)

## Experimental Evaluation

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

## Experimental Execution Report (Sklearn vs Custom)

Rationale: Since the SGD from SKlearn is already tested by the community we will use its executions as a baseline to compare its performance with our custom implementation.

- (Sklearn) GridSearch for HyperParameters by K-Fold Cross Validation
  - Learning rate Values
  - Learning rate Regularization
  - Number Iterations
- (Custom) Apply the parameters found on item (b), and compare the results.

- Consolidate the results from the 3 steps above into the final report.

