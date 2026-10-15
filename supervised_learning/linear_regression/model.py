import numpy as np

class LinearRegression:

    def __init__(self, learning_rate=0.01, iterations=1200):
        self.learning_rate = learning_rate
        self.iterations = iterations
        self.w = None
        self.b = 0

    def fit(self, X, y):
        m, n = X.shape

        self.w = np.zeros(n)
        self.b = 0

        for i in range(self.iterations):

            error = self.compute_error(X, y)

            dj_db = np.sum(error) / m
            dj_dw = (1 / m) * (X.T @ error)

            self.w = self.w - self.learning_rate * dj_dw
            self.b = self.b - self.learning_rate * dj_db
            if i == 1000 :
                print(f"dj_dw = {dj_dw}")
            if i % 100 == 0 :
                print(f"Iteration number {i} and cost function at this iteration = {self.compute_cost(X , y)}")

    def predict(self, X):
        return X @ self.w + self.b

    def compute_error(self, X, y):
        y_predict = self.predict(X)
        return y_predict - y
    
    def compute_cost(self , X , y) :
        sum_error = np.sum(self.compute_error(X , y) ** 2)
        return sum_error / (2 * len(X))