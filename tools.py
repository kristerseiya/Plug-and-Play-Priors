
from scipy.fftpack import dct, idct
import numpy as np
from PIL import Image
from scipy.signal import fftconvolve as convolve
from scipy.fftpack import fft, ifft
import matplotlib.pyplot as plt

def idct2d(x):
    return idct(idct(x, axis=0, norm='ortho'), axis=1, norm='ortho')

def dct2d(x):
    return dct(dct(x, axis=0, norm='ortho'), axis=1, norm='ortho')

def ifft2d(x):
    return ifft(ifft(x, axis=0), axis=1)

def fft2d(x):
    return fft(fft(x, axis=0), axis=1)

def image2uint8(img):
    if img.dtype != np.uint8:
        img = img * 255
        img = np.clip(img, 0, 255)
        img = img.astype(np.uint8)
    return img

def stackview(imgs, width=20, method='matplotlib'):
    h = imgs[0].shape[0]
    sep = np.zeros([h, width], dtype=np.uint8)
    view = image2uint8(imgs[0])
    for img in imgs[1:]:
        view = np.concatenate([view, sep, image2uint8(img)], axis=1)
    if method == 'Pillow':
        view = Image.fromarray(view, 'L')
        view.show()
    elif method == 'matplotlib':
        plt.imshow(view, cmap='gray')
        plt.show()

def compute_mse(x, y):

    if x.dtype == np.uint8:
        scale = 255
    else:
        scale = 1

    mse = np.mean(np.power(x - y, 2))
    psnr = 10 * np.log10(scale**2 / mse)
    return mse, psnr

def get_gauss2d(h, w, sigma):
    gauss_1d_w = np.array([np.exp(-(x-w//2)**2/float(2**sigma**2)) for x in range(w)])
    gauss_1d_w = gauss_1d_w / gauss_1d_w.sum()
    gauss_1d_h = np.array([np.exp(-(x-h//2)**2/float(2**sigma**2)) for x in range(h)])
    gauss_1d_h = gauss_1d_h
    gauss_2d = np.array([gauss_1d_w * s for s in gauss_1d_h])
    gauss_2d = gauss_2d / gauss_2d.sum()
    return gauss_2d

def compute_ssim(img1, img2, window_size=11, sigma=1.5):

    if img1.dtype == np.uint8:
        scale = 255
    else:
        scale = 1

    gauss_2d = get_gauss2d(window_size, window_size, sigma)

    mu1 = convolve(img1, gauss_2d, mode='valid')
    mu2 = convolve(img2, gauss_2d, mode='valid')
    sigma1 = convolve(gauss_2d, img1 * img1, mode='valid') - np.power(mu1, 2)
    sigma2 = convolve(gauss_2d, img2 * img2, mode='valid') - np.power(mu2, 2)
    sigma12 = convolve(gauss_2d, img1 * img2, mode='valid') - (mu1 * mu2)

    k1 = 0.01
    k2 = 0.03
    c1 = (k1 * 255)**2
    c2 = (k2 * 255)**2

    numerator = (2 * mu1 * mu2 + c1) * (2 * sigma12 + c2)
    denominator = (np.power(mu1, 2) + np.power(mu2, 2) + c1) * (sigma1 + sigma2 + c2)
    return np.mean(numerator / denominator)

def transposed_correlate(x, window):
    M, N = window.shape
    assert(M % 2 == 1)
    assert(M % 2 == 1)
    mc = M // 2
    nc = N // 2
    x_M, x_N = x.shape
    result = np.zeros((x.shape[0], x.shape[1]))
    for m in range(M):
        for n in range(N):
            result += np.roll(x, (m-mc, n-nc), axis=(0, 1)) * window[m, n]
    return result
