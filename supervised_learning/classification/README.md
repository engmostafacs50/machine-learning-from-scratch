# Logistic Regression from Scratch

A simple implementation of **Logistic Regression** from scratch using Python and NumPy, without relying on machine learning libraries such as Scikit-learn.

This project focuses on understanding the mathematical foundations of Logistic Regression, including the Sigmoid Function, Logistic Cost Function, Gradient Calculation, and Gradient Descent.

##  Overview

Logistic Regression is a supervised machine learning algorithm used for classification problems.

Unlike Linear Regression, which predicts continuous values, Logistic Regression predicts probabilities that an example belongs to a particular class.

For binary classification, the output is converted into one of two classes:

* `0`: Negative class
* `1`: Positive class

##  Mathematical Foundations

### 1. Linear Function

First, calculate the linear combination of the input features:

$$
z = Xw + b
$$

Where:

* \(X\): Input features
* \(w\): Model weights
* \(b\): Bias

### 2. Sigmoid Function

The Sigmoid Function converts the linear output into a probability between 0 and 1.

$$
f_{w,b}(x) = \frac{1}{1 + e^{-z}}
$$

### 3. Logistic Cost Function

To measure the model's performance, we use Binary Cross-Entropy (Log Loss).

$$
J(w,b) = -\frac{1}{m}\sum_{i=1}^{m}
\left[
y^{(i)}\log(f^{(i)})
+
(1-y^{(i)})\log(1-f^{(i)})
\right]
$$

Where:

* \(m\): Number of training examples
* \(y^{(i)}\): Actual label
* \(f^{(i)}\): Predicted probability

### 4. Gradient Calculation

The gradients determine how the weights and bias should change to minimize the cost.

**Weight Gradient:**

$$
\frac{\partial J}{\partial w_j}
=
\frac{1}{m}\sum_{i=1}^{m}
(f^{(i)}-y^{(i)})x_j^{(i)}
$$

**Bias Gradient:**

$$
\frac{\partial J}{\partial b}
=
\frac{1}{m}\sum_{i=1}^{m}
(f^{(i)}-y^{(i)})
$$

### 5. Gradient Descent

We update the model parameters iteratively:

$$
w_j = w_j - \alpha \frac{\partial J}{\partial w_j}
$$

$$
b = b - \alpha \frac{\partial J}{\partial b}
$$

Where \(\alpha\) represents the learning rate.

##  Implementation

The model is implemented using NumPy and includes the following methods:

| Method                  | Description                                                 |
| ----------------------- | ----------------------------------------------------------- |
| `__init__()`            | Initializes learning rate and number of iterations.         |
| `z()`                   | Calculates the linear combination of features and weights.  |
| `sigmoid()`             | Converts linear outputs into probabilities.                 |
| `predict_probability()` | Returns predicted probabilities.                            |
| `compute_error()`       | Calculates prediction errors.                               |
| `compute_cost()`        | Calculates Binary Cross-Entropy cost.                       |
| `compute_gradient()`    | Calculates gradients for weights and bias.                  |
| `gradient_descent()`    | Updates model parameters iteratively.                       |
| `fit()`                 | Initializes parameters and trains the model.                |
| `predict()`             | Converts probabilities into class labels using a threshold. |

## Project Structure

```text
logistic_regression/
│
├── README.md
├── model.py
└── main.py
```


##  Technologies

* Python
* NumPy

## Limitations

* Supports binary classification.
* Does not currently implement regularization.
* Does not include automatic feature scaling.
* Performance depends on the learning rate and number of iterations.

## Learning Objectives

Through this implementation, I practiced:

* Understanding Logistic Regression mathematically.
* Implementing the Sigmoid Function.
* Understanding Binary Cross-Entropy.
* Calculating gradients using vectorized NumPy operations.
* Implementing Gradient Descent.
* Training and evaluating a classification model from scratch.

---

**Part of my Machine Learning From Scratch journey.**
