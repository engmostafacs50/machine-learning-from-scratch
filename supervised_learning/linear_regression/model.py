import numpy as np
# remaining apply feuture scaling 
# remaining apply choosing suitable learning rate 
class LinearRegression:

    def __init__(self,  learning_rate=0.01, iterations=20000):
        self.learning_rate = learning_rate
        self.iterations = iterations
        self.w = None
        self.b = 0

    def fit(self, X, y):
        m, n = X.shape

        self.w = np.zeros(n)
        self.b = 0

        previous_cost = None
        
        for i in range(self.iterations):

            error = self.compute_error(X, y)

            dj_db = np.sum(error) / m
            dj_dw = (1 / m) * (X.T @ error)

            self.w = self.w - self.learning_rate * dj_dw
            self.b = self.b - self.learning_rate * dj_db
            
            current_cost = self.compute_cost(X , y)
            # if i % 100 == 0:
            #     print(f"Iteration {i}: Cost = {current_cost}")

            if previous_cost is not None:
                if abs(current_cost - previous_cost) <= 0.0001:
                    print(f"Converged at iteration {i + 1}")
                    # print(f"Final Cost: {current_cost}")
                    break
            previous_cost = current_cost
        print(f"Final Training Cost: {self.compute_cost(X, y)}")
        
            
    def predict(self, X):
        return X @ self.w + self.b

    def compute_error(self, X, y):
        y_predict = self.predict(X)
        return y_predict - y
    
    def compute_cost(self , X , y) :
        sum_error = np.sum(self.compute_error(X , y) ** 2)
        return sum_error / (2 * len(X))