import numpy as np
from model import LogisticRegression

X = np.array([
    [1],
    [2],
    [3],
    [4],
    [5],
    [6],
    [7],
    [8]
])

y = np.array([0, 0, 0, 0, 1, 1, 1, 1])

model = LogisticRegression(
    learning_rate=0.01,
    iterations=10000
)

# Initial parameters
model.w = np.zeros(X.shape[1])
model.b = 0.0

initial_cost = model.compute_cost(X, y)

# Training
model.fit(X, y)

final_cost = model.compute_cost(X, y)

print("Initial Cost:", initial_cost)
print("Final Cost:", final_cost)

print("Weights:", model.w)
print("Bias:", model.b)

print('#' * 100)

# Test Probablities 

probabilities = model.predict_probability(X)
predictions = model.predict(X)

print("Probabilities:", probabilities)
print("Predictions:", predictions) # I put threshold = 0.5
print("Actual Labels:", y) 

print('#' * 100) 
print("***** Evaluate Accuracy *****")

accuracy = np.mean(predictions == y) * 100

print(f"Accuracy: {accuracy:.2f}%")

print("***** Test unseen data *****")
X_new = np.array([
    [2.5],
    [4.5],
    [6.5]
])

print("New Probabilities:", model.predict_probability(X_new))
print("New Predictions:", model.predict(X_new))