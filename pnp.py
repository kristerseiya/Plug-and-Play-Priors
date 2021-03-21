
import numpy as np


class IdentityTransform:
    def __call__(self, x):
        return x

    def inverse(self, v):
        return v

class PnP_ADMM:
    def __init__(self, forward, image_prior, transform=None):
        self.forward = forward
        self.prior = image_prior
        self.transform = transform if transform != None else IdentityTransform()

    def run(self, iter=100, verbose=True,  relax=1., return_value='both'):

        v = np.zeros(self.prior.input_shape)
        u = np.zeros(self.prior.input_shape)

        for i in range(iter):

            if verbose:
                print('Iteration #{:d}'.format(i+1))

            x = self.forward(self.transform.inverse(v-u))

            x_relaxed = relax * self.transform(x) + (1 - relax) * v
            v = self.prior(x_relaxed+u)

            diff = x_relaxed - v
            u = u + diff

            if verbose:
                print('Difference: {:.5f}'.format(np.linalg.norm(diff, 1)))

        if return_value == 'both':
            return x, v
        elif return_value == 'x':
            return x
        return v
