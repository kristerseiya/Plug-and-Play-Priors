
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
    def __init__(self, lambd):
        self.lambd = lambd

    def __call__(self, x):
        return np.maximum(np.abs(x) - self.lambd, 0) * np.sign(x)

class TVNorm:
    def __init__(self, lambd):
        self.lambd = lambd

    def __call__(self, x):
        return prx_tv(x, self.lambd)

class DnCNN_Prior:
    def __init__(self, model_path, use_tensor=False,
                 patch_size=-1, device=None):

        if device == None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = device

        self.net = load_dncnn(model_path, device=device)
        self.patch_size = patch_size
        self.use_tensor = use_tensor

    def __call__(self, x):
        if not self.use_tensor:
            x = torch.tensor(x, dtype=torch.float32,
                             requires_grad=False, device=self.device)
            if self.patch_size > 0:
                x = x.view(batch_size, 1, self.patch_size, self.patch_size)
            else:
                x = x.view(1, 1, *x.size())
            y = self.net(x)
            y = y.cpu().squeeze(0).squeeze(0)
            y = y.numpy()
            return y
        return self.net(x)

class BM3D_Prior:
    def __init__(self, lambd):
        self.std = np.sqrt(lambd / 2)

    def __call__(self, x):
        return pybm3d.bm3d.bm3d(x, self.std)

# 1/2*alpha*|| y - x ||_2^2
class MSE:
    def __init__(self, y, alpha=1.):
        self.y = y
        self.alpha = alpha

    def __call__(self, x):
        return (self.alpha*self.y + x) / (self.alpha + 1)

# 1/2*alpha*|| y - Ax ||_2^2
class MSE2:
    def __init__(self, A, y, alpha=1.):
        self.y = y
        self.Aty = A.T @ y
        self.AtA = A.T @ A
        I = np.eye(self.AtA.shape[0])
        self.aAtA_inv = la.inv(I + self.alpha * self.AtA)

    def __call__(self, x):
        return self.aAtA_inv @ (self.alpha * self.Aty + x)

# mse for compressed sensing
class MSEwithMask:
    def __init__(self, y, mask, alpha=1.):
        self.y = y.copy()
        self.y[~mask] = 0.
        self.mask = mask
        self.alpha = alpha
        self.a = np.ones(self.mask.shape)
        self.a[self.mask] = 1. / (1 + alpha)

    def __call__(self, x):
        return self.a * (self.alpha * self.y + x)

class MSEwithMaskTensor:
    def __init__(self, y, mask, alpha=1.):
        self.y = y.clone()
        self.y[~mask] = 0.
        self.mask = mask
        self.alpha = alpha
        self.a = torch.ones_like(self.mask).type(torch.float32)
        self.a[self.mask] = 1. / (1 + alpha)

    def __call__(self, x):
        return self.a * (self.alpha * self.y + x)
