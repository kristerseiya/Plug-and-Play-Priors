
import numpy as np
import numpy.linalg as la
from prox_tv import tv1_2d as prx_tv
import torch
import torch.nn.functional as F
from scipy.signal import correlate2d
from scipy.signal import fftconvolve

from DnCNN.utils import load_dncnn
import tools


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
        return (self.y + self.alpha*x) / (1 + self.alpha)

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

# mse for deblurring
class MSEwithCorrelation:
    def __init__(self, y, window, input_shape):
        self.y = tools.transposed_correlate(y, window)
        self.window = window
        self.pad_size = ((window.shape[0]//2, window.shape[0]//2), (window.shape[1]//2, window.shape[1]//2))
        autocorr = correlate2d(window, window, mode='full')
        self.input_shape = input_shape
        shift = (autocorr.shape[0] // 2, autocorr.shape[1] // 2)
        autocorr = np.pad(autocorr, ((0, input_shape[0]-window.shape[0]), (0, input_shape[1]-window.shape[1])), 'constant', constant_values=((0, 0), (0, 0)))
        self.autocorr = np.roll(autocorr, (-shift[0], -shift[1]), axis=(0, 1))


    def set(self, alpha):
        self.alpha = alpha
        self.autocorr[0, 0] += alpha
        self.inv_window = tools.fft2d(self.autocorr)

    def prox(self, x):
        tmp = self.y + self.alpha * np.pad(x, self.pad_size, 'constant', constant_values=((0, 0), (0, 0)))
        tmp = tools.fft2d(tmp)
        result = np.real(tools.ifft2d(tmp / self.inv_window))
        return result[self.pad_size[0][0]:-self.pad_size[0][1], self.pad_size[1][0]:-self.pad_size[1][1]]
        # return correlate2d(self.y + self.alpha*x, self.inv_window, mode='valid')
