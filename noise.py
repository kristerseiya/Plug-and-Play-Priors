
import numpy as np

def add_gauss(image, std=0.1):
    return image + np.random.normal(size=image.shape) * std

def add_poisson(image, lambd=70.):
    # image = image*255
    # lambd = len(np.unique(image.astype(np.uint8)))
    # print(lambd)
    # lambd = 2 ** np.ceil(np.log2(lambd))
    return np.random.poisson(image * lambd) / float(lambd)
