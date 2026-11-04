import numpy as np 
class LogisticRegression:

    def __init__(self, learning_rate=0.01, iterations=10000):
        self.learning_rate = learning_rate
        self.iterations = iterations
        self.w = None 
        self.b = 0 

    def z(self, X):
        z = X @ self.w + self.b
        return z

    def sigmoid(self, z):
        s = 1 / (1 + np.exp(-z)) 
        return s 
    def predict_probability(self, X):
        z = self.z(X)
        predict_probability = self.sigmoid(z)
        return predict_probability

    def compute_error(self, X, y):
        error = self.predict_probability(X) - y
        return error 
        

    def compute_cost(self, X, y):
        m = X.shape[0] 
        predict_probability = self.predict_probability(X)
        
        predict_probability = np.clip(
        predict_probability, 1e-15, 1 - 1e-15
        )
        error_sum = np.sum(((y * np.log(predict_probability)) + (1 - y) * np.log(1 - predict_probability)))
        return (-1/m) * (error_sum) 
    
    def compute_gradient(self, X, y):
        m = X.shape[0]
        error = self.compute_error(X , y)
        dj_db = np.sum(error) / m
        dj_dw = (1 / m) * (X.T @ error)
        
        return dj_dw , dj_db

    def gradient_descent(self, X, y):
        previous_cost = None
        for i in range(self.iterations) :
            dj_dw , dj_db = self.compute_gradient(X , y)
            self.w = self.w - self.learning_rate * dj_dw
            self.b = self.b - self.learning_rate * dj_db
            current_cost = self.compute_cost(X , y)
            
            if previous_cost is not None:
                     if abs(current_cost - previous_cost) <= 0.0001:
                        print(f"Converged at iteration {i + 1}")
                        break
                                
            previous_cost = current_cost

    def fit(self, X, y):
        m , n = X.shape
        self.w = np.zeros((n , ))
        self.b = 0.0
        self.gradient_descent(X , y)
        
    def predict(self, X, threshold=0.5):
        predict_probability = self.predict_probability(X) 
        return (predict_probability >= threshold).astype(int)