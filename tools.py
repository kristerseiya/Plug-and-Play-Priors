
from scipy.fftpack import dct, idct
import numpy as np
from PIL import Image

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
