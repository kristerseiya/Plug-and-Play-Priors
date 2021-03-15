
import numpy as np
import numpy.linalg as la


# lambd * ||x||_1^2
class L1Norm:
    def __init__(self, lambd):
        self.lambd = lambd

    def set(self, alpha):
        self.alpha = alpha

    def prox(self, x):
        return np.maximum(np.abs(x) - self.lambd / self.alpha, 0) * np.sign(x)

# 1/2*|| y - Ax ||_2^2
class MSE:
    def __init__(self, A, y):
        self.y = y
        self.Aty = A.T @ y
        self.AtA = A.T @ A

    def set(self, alpha):
        alphaI = alpha * np.eye(self.AtA.shape[0])
        self.aAtA_inv = la.inv(alphaI + self.AtA)

    def prox(self, x):
        return self.aAtA_inv @ (self.Aty + alpha*x)

# mse for compressed sensing
class MSE_CS:
    def __init__(self, y, mask):
        self.y = y
        self.y[~mask] = 0.
        self.mask = mask

    def set(self, alpha):
        self.alpha = alpha
        self.a = np.ones(self.mask.shape)
        self.a[self.mask] = 1. / (1 + alpha)
        self.a[~self.mask] = 1. / alpha

    def prox(self, x):
        return self.a * (self.y + self.alpha*x)
