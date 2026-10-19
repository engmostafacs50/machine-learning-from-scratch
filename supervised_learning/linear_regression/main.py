import numpy as np
from model import LinearRegression
def z_score_normalize_features(X) :
    mu = np.mean(X , axis = 0)
    segma = np.std(X , axis = 0)
    X_norm = (X - mu) / segma
    
    return mu , segma , X_norm

X = np.array([
    [1000, 5],
    [1200, 10],
    [1500, 8],
    [1800, 15],
    [2000, 4],
    [2200, 12],
    [2500, 6],
    [2800, 20],
    [3000, 10],
    [3500, 7],
    [1100, 18],
    [1600, 25],
    [2100, 16],
    [2600, 14],
    [3200, 3],
    [1400, 9],
    [1900, 22],
    [2400, 11],
    [2900, 5],
    [3300, 17]
], dtype=float)


y = np.array([
    150,
    180,
    220,
    240,
    300,
    280,
    350,
    330,
    400,
    450,
    130,
    170,
    260,
    320,
    480,
    200,
    210,
    340,
    430,
    390
], dtype=float)


rng = np.random.default_rng(42)
indices = rng.permutation(len(X))

split = int(len(X) * 0.8)

train_indices = indices[:split]
test_indices = indices[split:]

X_train = X[train_indices]
y_train = y[train_indices]

X_test = X[test_indices]
y_test = y[test_indices]

X_mu , X_sigma , X_train_norm  = z_score_normalize_features(X_train)
X_test_norm = (X_test - X_mu) / X_sigma

model = LinearRegression(
     learning_rate=0.001,
    iterations=10000
)



model.fit(X_train_norm, y_train)


y_pred = model.predict(X_test_norm)

mae = np.mean(np.abs(y_test - y_pred))
mse = np.mean((y_test - y_pred) ** 2)
print("Weights:", model.w)
print("Bias:", model.b)
print("\nActual Prices:", y_test)
print("Predicted Prices:", y_pred)
print("\nMean Absolute Error:", mae)
print("Mean Squared Error:", mse)