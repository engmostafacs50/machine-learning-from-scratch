# Linear Regression From Scratch

A simple implementation of Linear Regression using Python and NumPy, built from scratch to understand the mathematical foundations and optimization process behind Machine Learning models.

## Overview

This project implements a Linear Regression model using **Batch Gradient Descent** to predict continuous numerical values from multiple input features.

The implementation uses NumPy for numerical computations without relying on machine learning libraries such as Scikit-learn.

## Project Structure

```text
linear_regression/
│
├── model.py
├── main.py
└── README.md
```

* `model.py`: Contains the Linear Regression implementation.
* `main.py`: Handles data preparation, training, predictions, and evaluation.

## Mathematical Background

### 1. Hypothesis Function

The model predicts an output using:

$$
f_{w,b}(x)=w\cdot x+b
$$

Where:

* `w`: Weight parameters.
* `b`: Bias parameter.
* `x`: Input features.

### 2. Cost Function

Mean Squared Error Cost Function:

$$
J(w,b)=\frac{1}{2m}\sum_{i=0}^{m-1}(f_{w,b}(x^{(i)})-y^{(i)})^2
$$

It measures how far the model's predictions are from the actual target values.

### 3. Gradient Descent

The model uses Batch Gradient Descent to minimize the cost function.

Weight update:

$$
w_j=w_j-\alpha\frac{\partial J}{\partial w_j}
$$

Bias update:

$$
b=b-\alpha\frac{\partial J}{\partial b}
$$

Where:

* `α`: Learning rate.
* `m`: Number of training examples.

The gradients are calculated using:

$$
dj\_dw=\frac{1}{m}X^T(error)
$$

$$
dj\_db=\frac{1}{m}\sum error
$$

## Implementation Details

The `LinearRegression` class provides the following methods:

| Method            | Description                                     |
| ----------------- | ----------------------------------------------- |
| `__init__()`      | Initializes learning rate and iterations.       |
| `fit()`           | Trains the model using gradient descent.        |
| `predict()`       | Generates predictions using learned parameters. |
| `compute_error()` | Calculates prediction errors.                   |
| `compute_cost()`  | Calculates the cost function.                   |

### Training Optimization

The implementation includes an early stopping condition.

Training stops when the absolute difference between consecutive cost values becomes smaller than or equal to `0.0001`.

## Data Preparation

The main script performs the following steps:

1. Creates a numerical dataset with two input features.
2. Splits the dataset into training and testing sets (80/20).
3. Applies Z-score normalization to the training features.
4. Uses the training mean and standard deviation to normalize the testing features.
5. Trains the model using the normalized training data.

### Z-Score Normalization

$$
x_{norm}=\frac{x-\mu}{\sigma}
$$

Where:

* `μ`: Feature mean.
* `σ`: Feature standard deviation.

**Important:** The normalization parameters calculated from the training set are reused for the test set to prevent data leakage.

## Model Evaluation

After training, the model predicts values for the test dataset.

Performance is evaluated using:

**1. Mean Absolute Error (MAE)**

$$
MAE=\frac{1}{m}\sum |y-\hat y|
$$

**2. Mean Squared Error (MSE)**

$$
MSE=\frac{1}{m}\sum(y-\hat y)^2
$$

These metrics help measure the difference between actual and predicted values.

## Technologies

* Python
* NumPy

## Current Limitations

* No automatic feature selection.
* Learning rate is manually specified.
* Feature scaling is handled outside the model class.
* No regularization is implemented yet.

## Future Improvements

* Experiment with different learning rates.
* Implement automatic feature scaling.
* Explore polynomial regression.
* Implement regularization (Ridge Regression).
* Compare model performance with Scikit-learn.

## Learning Objectives

This project was developed to gain a deeper understanding of:

* Linear Regression fundamentals.
* Cost Function and optimization.
* Batch Gradient Descent.
* Vectorized NumPy operations.
* Feature scaling.
* Model evaluation.
