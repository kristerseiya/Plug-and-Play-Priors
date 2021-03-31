
import numpy as np
import numpy.linalg as la
from prox_tv import tv1_2d as prx_tv
import torch
import torch.nn.functional as F
from scipy.signal import correlate2d
from scipy.signal import fftconvolve
import pybm3d

from DnCNN.utils import load_dncnn
import tools


# lambd * ||x||_1
class L1Norm:
    def __init__(self, lambd, input_shape):
        self.lambd = lambd
        self.input_shape = input_shape

    def __call__(self, x):
        return np.maximum(np.abs(x) - self.lambd, 0) * np.sign(x)

class TVNorm:
    def __init__(self, lambd, input_shape):
        self.lambd = lambd
        self.input_shape = input_shape

    def __call__(self, x):
        return prx_tv(x, self.lambd)

class DnCNN_Prior:
    def __init__(self, model_path, input_shape, device=None):

        if device == None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = device

        self.net = load_dncnn(model_path, device=device)
        self.input_shape = input_shape

    def __call__(self, x):
        x = torch.tensor(x.transpose([1, 0]), dtype=torch.float32, requires_grad=False, device=self.device)
        x = x.view(1, 1, *x.size())
        y = self.net(x)
        y = y.cpu()
        y = y.view(y.size(-2), y.size(-1))
        y = y.numpy()
        y = y.transpose([1, 0])
        return y

class BM3D_Prior:
    def __init__(self, lambd, input_shape):
        self.std = np.sqrt(lambd / 2)
        self.input_shape = input_shape

    def __call__(self, x):
        return pybm3d.bm3d.bm3d(x, self.std)

# 1/2*alpha*|| y - x ||_2^2
class MSE:
    def __init__(self, y, input_shape, alpha=1.):
        self.y = y
        self.input_shape = input_shape
        self.alpha = alpha

    def __call__(self, x):
        return (self.alpha*self.y + x) / (self.alpha + 1)

# 1/2*alpha*|| y - Ax ||_2^2
class MSE2:
    def __init__(self, A, y, input_shape, alpha=1.):
        self.y = y
        self.Aty = A.T @ y
        self.AtA = A.T @ A
        self.input_shape = input_shape
        I = np.eye(self.AtA.shape[0])
        self.aAtA_inv = la.inv(I + self.alpha * self.AtA)

    def __call__(self, x):
        return self.aAtA_inv @ (self.alpha * self.Aty + x)

# mse for compressed sensing
class MSEwithMask:
    def __init__(self, y, mask, input_shape, alpha=1.):
        self.y = y.copy()
        self.y[~mask] = 0.
        self.mask = mask
        self.input_shape = input_shape
        self.alpha = alpha
        self.a = np.ones(self.mask.shape)
        self.a[self.mask] = 1. / (1 + alpha)

    def __call__(self, x):
        return self.a * (self.alpha * self.y + x)
