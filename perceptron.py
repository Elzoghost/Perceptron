import numpy as np

class Perceptron:
    def __init__(self, eta0=0.1, max_iter=100, random_state=1):
        self.eta0 = eta0
        self.max_iter = max_iter
        self.random_state = random_state
   
    def fit(self, X, y):
        
        rgen = np.random.RandomState(self.random_state)
        self.w_ = rgen.normal(loc=0.0, scale=0.01, size=1 + X.shape[1])
        self.errors_ = []
   
        for _ in range(self.max_iter):
            
            errors = 0
            for xi, yi in zip(X, y):
                update = self.eta0 * (yi - self.predict(xi))
                self.w_[1:] += update * xi
                self.w_[0] += update
                errors += int(update != 0.0)
            self.errors_.append(errors)
            if errors == 0:
                break
        return self

    def net_input(self, X):
        return np.dot(X, self.w_[1:]) + self.w_[0]

    def predict(self, X):
        return np.where(self.net_input(X) >= 0.0, 1, -1)
