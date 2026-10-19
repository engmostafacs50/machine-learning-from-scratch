import numpy as np
from model import LinearRegression

X = np.array([
    [50],
    [60],
    [70],
    [80],
    [90],
    [100],
    [110],
    [120],
    [130],
    [140]
])


y = np.array([
    100,
    120,
    140,
    160,
    180,
    200,
    220,
    240,
    260,
    280
])

X_train = X[:8]
y_train = y[:8]


X_test = X[8:]
y_test = y[8:]


model = LinearRegression(
     learning_rate=0.00001,
    iterations=1000
)



model.fit(X_train, y_train)


y_pred = model.predict(X_test)

mae = np.mean(np.abs(y_test - y_pred))
mse = np.mean((y_test - y_pred) ** 2)
print("Weights:", model.w)
print("Bias:", model.b)
print("\nActual Prices:", y_test)
print("Predicted Prices:", y_pred)
print("\nMean Absolute Error:", mae)
print("Mean Squared Error:", mse)