
import numpy as np
import numpy.linalg as la
from prox_tv import tv1_2d as prx_tv
from DnCNN.utils import load_dncnn
import torch
import torch.nn.functional as F

# lambd * ||x||_1
class L1Norm:
    def __init__(self, lambd, input_shape):
        self.lambd = lambd
        self.input_shape = input_shape

    def set(self, alpha):
        self.alpha = alpha

    def prox(self, x):
        return np.maximum(np.abs(x) - self.lambd / self.alpha, 0) * np.sign(x)

class TVNorm:
    def __init__(self, lambd, input_shape):
        self.lambd = lambd
        self.input_shape = input_shape

    def set(self, alpha):
        self.alpha = alpha

    def prox(self, x):
        return prx_tv(x, self.lambd / self.alpha)

class DnCNN_Prior:
    def __init__(self, model_path, input_shape):
        self.net = load_dncnn(model_path)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.input_shape = input_shape

    def set(self, alpha):
        pass

    def prox(self, x):
        x = torch.tensor(x.transpose([1, 0]), dtype=torch.float32, requires_grad=False, device=self.device)
        x = x.view(1, 1, *x.size())
        y = self.net(x)
        y = y.cpu()
        y = y.view(y.size(-2), y.size(-1))
        y = y.numpy()
        y = y.transpose([1, 0])
        return y

# 1/2*|| y - x ||_2^2
class MSE:
    def __init__(self, y, input_shape):
        self.y = y
        self.input_shape = input_shape

    def set(self, alpha):
        self.alpha = alpha

    def prox(self, x):
        return (self.y + alpha*x) / (1 + self.alpha)

# 1/2*|| y - Ax ||_2^2
class MSE2:
    def __init__(self, A, y, input_shape):
        self.y = y
        self.Aty = A.T @ y
        self.AtA = A.T @ A
        self.input_shape = input_shape

    def set(self, alpha):
        alphaI = alpha * np.eye(self.AtA.shape[0])
        self.aAtA_inv = la.inv(alphaI + self.AtA)

    def prox(self, x):
        return self.aAtA_inv @ (self.Aty + alpha*x)

# mse for compressed sensing
class MSEwithMask:
    def __init__(self, y, mask, input_shape):
        self.y = y.copy()
        self.y[~mask] = 0.
        self.mask = mask
        self.input_shape = input_shape

    def set(self, alpha):
        self.scale = alpha
        self.a = np.ones(self.mask.shape)
        self.a[self.mask] = 1. / (1 + alpha)
        self.a[~self.mask] = 1. / alpha

    def prox(self, x):
        return self.a * (self.y + self.scale*x)

# mse for compressed sensing
class MSEwithConvolve:
    def __init__(self, y, window, input_shape):
        self.y = y.copy()
        self.y[~mask] = 0.
        self.mask = mask
        self.input_shape = input_shape

    def set(self, alpha):
        self.scale = alpha
        self.a = np.ones(self.mask.shape)
        self.a[self.mask] = 1. / (1 + alpha)
        self.a[~self.mask] = 1. / alpha

    def prox(self, x):
        return self.a * (self.y + self.scale*x)
