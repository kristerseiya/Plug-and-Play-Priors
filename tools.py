
from scipy.fftpack import dct, idct
import numpy as np

def idct2d(x):
    return idct(idct(x, axis=0, norm='ortho'), axis=1, norm='ortho')

def dct2d(x):
    return dct(dct(x, axis=0, norm='ortho'), axis=1, norm='ortho')

def image_reformat(img):
    img = img * 255
    img = np.clip(img, 0, 255)
    img = img.astype(np.uint8)
    return img
