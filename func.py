
import numpy as np
import numpy.linalg as la
import torch
import torch.nn.functional as F
from scipy.signal import correlate2d
from scipy.signal import fftconvolve

try:
    from prox_tv import tv1_2d as prx_tv
    import pybm3d
except:
    pass

from DnCNN.utils import load_dncnn
import tools


# lambd * ||x||_1
class L1Norm:
    def __init__(self, lambd):
        self.lambd = lambd

    def prox(self, x):
        return np.maximum(np.abs(x) - self.lambd, 0) * np.sign(x)

class ProxTV:
    def __init__(self, lambd):
        self.lambd = lambd

    def prox(self, x):
        return prx_tv(x, self.lambd)

class DnCNN:
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

class BM3D:
    def __init__(self, sigma):
        self.sigma = sigma

    def __call__(self, x):
        return pybm3d.bm3d.bm3d(x, self.sigma)

# 1/2*alpha*|| y - x ||_2^2
class MSE:
    def __init__(self, y, alpha=1.):
        self.y = y
        self.alpha = alpha

    def prox(self, x):
        return (self.alpha*self.y + x) / (self.alpha + 1)

# 1/2*alpha*|| y - Ax ||_2^2
class MSE2:
    def __init__(self, A, y, alpha=1.):
        self.y = y
        self.Aty = A.T @ y
        self.AtA = A.T @ A
        I = np.eye(self.AtA.shape[0])
        self.aAtA_inv = la.inv(I + self.alpha * self.AtA)

    def prox(self, x):
        return self.aAtA_inv @ (self.alpha * self.Aty + x)

# mse for compressed sensing
class MaskMSE:
    def __init__(self, y, mask, alpha=1.):
        self.y = y.copy()
        self.y[~mask] = 0.
        self.mask = mask
        self.alpha = alpha
        self.a = np.ones(self.mask.shape)
        self.a[self.mask] = 1. / (1 + alpha)

    def prox(self, x):
        return self.a * (self.alpha * self.y + x)

class MaskMSETensor:
    def __init__(self, y, mask, alpha=1.):
        self.y = y.clone()
        self.y[~mask] = 0.
        self.mask = mask
        self.alpha = alpha
        self.a = torch.ones_like(self.mask).type(torch.float32)
        self.a[self.mask] = 1. / (1 + alpha)

    def prox(self, x):
        return self.a * (self.alpha * self.y + x)

# mse for deblurring
class CirculantMSE:
    def __init__(self, y, window, input_shape, alpha=1.):
        self.Hty = tools.transposed_correlate(y, window)
        self.window = window
        autocorr = correlate2d(window, window, mode='full')
        self.input_shape = input_shape
        shift = (window.shape[0]-1, window.shape[1]-1)
        autocorr = np.pad(autocorr, ((0, input_shape[0]-autocorr.shape[0]), (0, input_shape[1]-autocorr.shape[1])), 'constant', constant_values=((0, 0), (0, 0)))
        self.autocorr = alpha * np.roll(autocorr, (-shift[0], -shift[1]), axis=(0, 1))
        self.alpha = alpha
        self.autocorr[0, 0] += 1
        self.inv_window = tools.fft2d(self.autocorr)

    def prox(self, x):
        tmp = self.alpha * self.Hty + x
        tmp = tools.fft2d(tmp)
        result = np.real(tools.ifft2d(tmp / self.inv_window))
        return result

class AverageDownsampleMSE:
    def __init__(self, y, size, alpha, input_shape):
        self.size = size
        self.alpha = alpha
        self.input_shape = input_shape
        self.size = size
        self.aAty = np.zeros(input_shape)
        M, N = self.input_shape
        for i, m in enumerate(range(0, M, size)):
            for j, n in enumerate(range(0, N, size)):
                step_m = size
                step_n = size
                if M - m < size:
                    step_m = M - m
                if N - n < size:
                    step_n = N - n
                self.aAty[m:m+step_m, n:n+step_n] = y[i, j] / (step_m * step_n)
        self.aAty = self.aAty * self.alpha

    def prox(self, x):
        x = x + self.aAty
        output = np.zeros_like(x)
        M, N = self.input_shape
        for m in range(0, M, self.size):
            for n in range(0, N, self.size):
                step_m = self.size
                step_n = self.size
                if M - m < self.size:
                    step_m = M - m
                if N - n < self.size:
                    step_n = N - n
                scale = self.alpha / ((step_m * step_n))
                output[m:m+step_m, n:n+step_n] = (x[m:m+step_m, n:n+step_n]).mean() * (scale / (1 + scale))
        output = x - output
        return output
