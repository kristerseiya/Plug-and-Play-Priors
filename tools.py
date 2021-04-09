
from scipy.fftpack import dct, idct
import numpy as np
from PIL import Image
from scipy.signal import fftconvolve as convolve
from scipy.fftpack import fft, ifft
import matplotlib.pyplot as plt
from scipy import ndimage

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

def compute_mse(x, y, scale=None):

    if scale == None:
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

# def compute_ssim(img1, img2, window_size=11, sigma=1.5):
#
#     if img1.dtype == np.uint8:
#         scale = 255
#     else:
#         scale = 1
#
#     gauss_2d = get_gauss2d(window_size, window_size, sigma)
#
#     mu1 = convolve(img1, gauss_2d, mode='valid')
#     mu2 = convolve(img2, gauss_2d, mode='valid')
#     sigma1 = convolve(gauss_2d, img1 * img1, mode='valid') - np.power(mu1, 2)
#     sigma2 = convolve(gauss_2d, img2 * img2, mode='valid') - np.power(mu2, 2)
#     sigma12 = convolve(gauss_2d, img1 * img2, mode='valid') - (mu1 * mu2)
#
#     k1 = 0.01
#     k2 = 0.03
#     c1 = (k1 * 255)**2
#     c2 = (k2 * 255)**2
#
#     numerator = (2 * mu1 * mu2 + c1) * (2 * sigma12 + c2)
#     denominator = (np.power(mu1, 2) + np.power(mu2, 2) + c1) * (sigma1 + sigma2 + c2)
#     return np.mean(numerator / denominator)

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

def fspecial_gauss(size, sigma):
    """Function to mimic the 'fspecial' gaussian MATLAB function
    """
    x, y = np.mgrid[-size//2 + 1:size//2 + 1, -size//2 + 1:size//2 + 1]
    g = np.exp(-((x**2 + y**2)/(2.0*sigma**2)))
    return g/g.sum()

def ssim(img1, img2, scale=None, cs_map=False):
    """Return the Structural Similarity Map corresponding to input images img1
    and img2 (images are assumed to be uint8)

    This function attempts to mimic precisely the functionality of ssim.m a
    MATLAB provided by the author's of SSIM
    https://ece.uwaterloo.ca/~z70wang/research/ssim/ssim_index.m
    """
    img1 = img1.astype(np.float64)
    img2 = img2.astype(np.float64)
    size = 11
    sigma = 1.5
    window = fspecial_gauss(size, sigma)
    K1 = 0.01
    K2 = 0.03
    if scale == None:
        if img1.dtype == np.uint8:
            L = 255
        else:
            L = 1
    else:
        L = scale
    # L = 255 #bitdepth of image
    C1 = (K1*L)**2
    C2 = (K2*L)**2
    mu1 = convolve(window, img1, mode='valid')
    mu2 = convolve(window, img2, mode='valid')
    mu1_sq = mu1*mu1
    mu2_sq = mu2*mu2
    mu1_mu2 = mu1*mu2
    sigma1_sq = convolve(window, img1*img1, mode='valid') - mu1_sq
    sigma2_sq = convolve(window, img2*img2, mode='valid') - mu2_sq
    sigma12 = convolve(window, img1*img2, mode='valid') - mu1_mu2
    if cs_map:
        return (((2*mu1_mu2 + C1)*(2*sigma12 + C2))/((mu1_sq + mu2_sq + C1)*
                    (sigma1_sq + sigma2_sq + C2)),
                (2.0*sigma12 + C2)/(sigma1_sq + sigma2_sq + C2))
    else:
        return ((2*mu1_mu2 + C1)*(2*sigma12 + C2))/((mu1_sq + mu2_sq + C1)*
                    (sigma1_sq + sigma2_sq + C2))

def msssim(img1, img2, scale=None):
    """This function implements Multi-Scale Structural Similarity (MSSSIM) Image
    Quality Assessment according to Z. Wang's "Multi-scale structural similarity
    for image quality assessment" Invited Paper, IEEE Asilomar Conference on
    Signals, Systems and Computers, Nov. 2003

    Author's MATLAB implementation:-
    http://www.cns.nyu.edu/~lcv/ssim/msssim.zip
    """
    if scale == None:
        if img1.dtype == np.uint8:
            L = 255
        else:
            L = 1
    else:
        L = scale
    level = 5
    weight = np.array([0.0448, 0.2856, 0.3001, 0.2363, 0.1333])
    downsample_filter = np.ones((2, 2))/4.0
    im1 = img1.astype(np.float64)
    im2 = img2.astype(np.float64)
    mssim = np.array([])
    mcs = np.array([])
    for l in range(level):
        ssim_map, cs_map = ssim(im1, im2, scale=L, cs_map=True)
        mssim = np.append(mssim, ssim_map.mean())
        mcs = np.append(mcs, cs_map.mean())
        filtered_im1 = ndimage.filters.convolve(im1, downsample_filter,
                                                mode='reflect')
        filtered_im2 = ndimage.filters.convolve(im2, downsample_filter,
                                                mode='reflect')
        im1 = filtered_im1[::2, ::2]
        im2 = filtered_im2[::2, ::2]
    return (np.prod(mcs[0:level-1]**weight[0:level-1])*
                    (mssim[level-1]**weight[level-1]))
