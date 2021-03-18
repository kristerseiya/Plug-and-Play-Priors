
from scipy.fftpack import dct, idct
import numpy as np
from PIL import Image
from scipy.signal import fftconvolve as convolve

def idct2d(x):
    return idct(idct(x, axis=0, norm='ortho'), axis=1, norm='ortho')

def dct2d(x):
    return dct(dct(x, axis=0, norm='ortho'), axis=1, norm='ortho')

def image_reformat(img):
    img = img * 255
    img = np.clip(img, 0, 255)
    img = img.astype(np.uint8)
    return img

def stackview(imgs, width=20):
    h = imgs[0].shape[0]
    sep = np.zeros([h, width], dtype=np.uint8)
    view = image_reformat(imgs[0])
    for img in imgs[1:]:
        view = np.concatenate([view, sep, image_reformat(img)], axis=1)
    view = Image.fromarray(view, 'L')
    view.show()

def compute_mse(x, y, reformat=True):
    if reformat:
        x = image_reformat(x)
        y = image_reformat(y)
    return np.mean(np.power(x - y, 2))

def compute_ssim(img1, img2, window_size=11, sigma=1.5, reformat=True):

    if reformat:
        img1 = image_reformat(img1)
        img2 = image_reformat(img2)

    gauss_1d = np.array([np.exp(-(x-window_size//2)**2/float(2**sigma**2)) for x in range(window_size)])
    gauss_1d = gauss_1d / gauss_1d.sum()
    gauss_2d = np.matmul(gauss_1d.reshape([-1, 1]), gauss_1d.reshape([1, -1]))

    mu1 = convolve(img1, gauss_2d, mode='valid')
    mu2 = convolve(img2, gauss_2d, mode='valid')
    sigma1 = convolve(img1 * img1, gauss_2d, mode='valid') - np.power(mu1, 2)
    sigma2 = convolve(img2 * img2, gauss_2d, mode='valid') - np.power(mu2, 2)
    sigma12 = convolve(img1 * img2, gauss_2d, mode='valid') - (mu1 * mu2)

    k1 = 0.01
    k2 = 0.03
    c1 = (k1 * 255)**2
    c2 = (k2 * 255)**2

    numerator = (2 * mu1 * mu2 + c1) * (2 * sigma12 + c2)
    denominator = (np.power(mu1, 2) + np.power(mu2, 2) + c1) * (sigma1 + sigma2 + c2)
    return np.mean(numerator / denominator)
