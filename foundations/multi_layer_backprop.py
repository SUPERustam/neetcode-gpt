import numpy as np
from typing import List


class Solution:
    def forward_and_backward(self,
                              x: List[float],
                              W1: List[List[float]], b1: List[float],
                              W2: List[List[float]], b2: List[float],
                              y_true: List[float]) -> dict:
        # Architecture: x -> Linear(W1, b1) -> ReLU -> Linear(W2, b2) -> predictions
        # Loss: MSE = mean((predictions - y_true)^2)
        #
        # Return dict with keys:
        #   'loss':  float (MSE loss, rounded to 4 decimals)
        #   'dW1':   2D list (gradient w.r.t. W1, rounded to 4 decimals)
        #   'db1':   1D list (gradient w.r.t. b1, rounded to 4 decimals)
        #   'dW2':   2D list (gradient w.r.t. W2, rounded to 4 decimals)
        #   'db2':   1D list (gradient w.r.t. b2, rounded to 4 decimals)
        W1 = np.array(W1)
        W2 = np.array(W2)
        b1 = np.array(b1)
        b2 = np.array(b2)
        x = np.array(x)
        y_true = np.array(y_true)
        n = y_true.shape[0]

        # forward pass
        z = x @ W1.T + b1
        a1 = np.maximum(0.0, z) # ReLU
        z2 = a1 @ W2.T + b2
        
        L = np.mean((y_true - z2) ** 2)

        # backward pass
        dL = 2 * (z2 - y_true) / n
        dW2 = np.outer(dL, a1)
        db2 = dL

        da1 = dL @ W2

        dz = da1 * (z > 0)
        # return {'loss': a1.shape, "dW1": dz.shape, "db1": 0,"dW2":0, "db2":0}

        dW1 = np.outer(dz, x)
        db1 = dz

        return {"loss": np.round(L, 4), "dW1": np.round(dW1, 4), "db1": np.round(db1, 4),"dW2": np.round(dW2, 4), "db2": np.round(db2, 4)}
