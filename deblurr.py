
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import argparse
from scipy.signal import correlate2d

import tools
import prox
import pnp
import noise

# command line arguments
parser = argparse.ArgumentParser()
parser.add_argument('--image', help='path to image', type=str, required=True)
parser.add_argument('--sigma', help='sample rate', type=float, default=1.5)
parser.add_argument('--iter', help='number of iteration', type=int, default=100)
parser.add_argument('--alpha', help='lagrange multiplier', type=float, default=1000)
parser.add_argument('--window', help='lagrange multiplier', type=int, default=15)
parser.add_argument('--weights', type=str, default=None)
args = parser.parse_args()

# read image
img = Image.open(args.image).convert('L')
img = np.array(img) / 255.
gauss_window = tools.get_gauss2d(args.window, args.window, args.sigma)
y = correlate2d(img, gauss_window, mode='same', boundary='wrap')

# mse for deblurring
class MSEwithCorrelation:
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

    def __call__(self, x):
        tmp = self.alpha * self.Hty + x
        tmp = tools.fft2d(tmp)
        result = np.real(tools.ifft2d(tmp / self.inv_window))
        return result

# forward model
mseloss = MSEwithCorrelation(y, gauss_window, input_shape=img.shape, alpha=args.alpha)

# prior
dncnn_prior = prox.DnCNN_Prior(args.weights, input_shape=img.shape)

# optimize
optimizer = pnp.PnP_ADMM(mseloss, dncnn_prior)
recon = optimizer.run(iter=args.iter, return_value='v')

# reconstruction quality assessment
mse, psnr = tools.compute_mse(img, recon)
ssim = tools.compute_ssim(img, recon)
print('MSE: {:.5f}'.format(mse))
print('PSNR: {:.5f}'.format(psnr))
print('SSIM: {:.5f}'.format(ssim))

# viewer
tools.stackview([img, y, recon], width=20, method='Pillow')
