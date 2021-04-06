
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image, ImageEnhance
import argparse
from scipy.signal import correlate2d

import tools
import prox
import pnp
import noise

# command line arguments
parser = argparse.ArgumentParser()
parser.add_argument('--image', help='path to image', type=str, required=True)
parser.add_argument('--iter', help='number of iteration', type=int, default=100)
parser.add_argument('--alpha', help='coeefficient of forward model', type=float, default=1000)
parser.add_argument('--size', help='pixel size', type=int, default=15)
parser.add_argument('--weights', type=str, default=None)
args = parser.parse_args()

def pixelate(x, size):
    M, N = x.shape
    output_shape = (int(np.ceil(M / size)), int(np.ceil(N / size)))
    output = np.zeros(output_shape)
    for i, m in enumerate(range(0, M, size)):
        for j, n in enumerate(range(0, N, size)):
            step_m = size
            step_n = size
            if M - m < size:
                step_m = M - m
            if N - n < size:
                step_n = N - n
            output[i, j] = (x[m:m+step_m, n:n+step_n]).mean()
    return output

class MSE_AverageSampling:
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

    def __call__(self, x):
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

# read image
img = Image.open(args.image).convert('L')
img = np.array(img) / 255.

y = pixelate(img, args.size)
forward = MSE_AverageSampling(y, args.size, args.alpha, img.shape)
prior = prox.DnCNN_Prior(args.weights, input_shape=img.shape)

# optimize
optimizer = pnp.PnP_ADMM(forward, prior)
recon = optimizer.run(iter=args.iter, return_value='v')

# reconstruction quality assessment
mse, psnr = tools.compute_mse(img, recon)
ssim = tools.compute_ssim(img, recon)
print('MSE: {:.5f}'.format(mse))
print('PSNR: {:.5f}'.format(psnr))
print('SSIM: {:.5f}'.format(ssim))

M, N = img.shape
size = args.size
y_big = np.zeros_like(img)
for i, m in enumerate(range(0, M, size)):
    for j, n in enumerate(range(0, N, size)):
        step_m = size
        step_n = size
        if M - m < size:
            step_m = M - m
        if N - n < size:
            step_n = N - n
        y_big[m:m+step_m, n:n+step_n] = y[i, j]

# viewer
tools.stackview([img, y_big, recon], width=20, method='Pillow')
