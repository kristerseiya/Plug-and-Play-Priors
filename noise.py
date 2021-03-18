
import numpy as np

def add_gauss(image, std=0.1):
    return image + np.random.normal(size=image.shape) * std

def add_poisson(image, lambd=70.):
    return np.random.poisson(image * lambd) / float(lambd)
