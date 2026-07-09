import numpy as np
from numpy.typing import NDArray
from typing import Tuple


class Solution:
    def train(self, X: NDArray[np.float64], y: NDArray[np.float64], epochs: int, lr: float) -> Tuple[NDArray[np.float64], float]:
        # X: (n_samples, n_features)
        # y: (n_samples,) targets
        # epochs: number of training iterations
        # lr: learning rate
        #
        # Model: y_hat = X @ w + b
        # Loss: MSE = (1/n) * sum((y_hat - y)^2)
        # Initialize w = zeros, b = 0
        
        w = np.zeros(X.shape[1])
        b = 0
        n = np.size(y)
        for epoch in range(epochs):
            y_hat = X @ w + b
            L = np.sum((y_hat - y) ** 2) / n

            dL_w = 2 * X.T @ (y_hat - y) / n
            # return (dL_w.shape, 0)
            dL_b = 2 * np.sum(y_hat - y) / n
            
            w -= lr * dL_w
            b -= lr * dL_b
        
        return (np.round(w, 5), round(b, 5))
        