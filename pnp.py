
import numpy as np

class IdentityTransform:
    def __call__(self, x):
        return x

    def inverse(self, v):
        return v

# transform(x) = v
# transform.inv(v) = x
class PnP_ADMM:
    def __init__(self, forward, image_prior, variable_shape, transform=None):
        self.forward = forward
        self.prior = image_prior
        self.variable_shape = variable_shape
        self.transform = transform if transform != None else IdentityTransform()

    def run(self, alpha=100, iter=100, verbose=True, return_value='both'):

        self.forward.set(alpha)
        self.prior.set(alpha)

        v = np.zeros(self.variable_shape)
        u = np.zeros(self.variable_shape)

        for i in range(iter):

            if verbose:
                print('Iteration #{:d}'.format(i+1))

            x = self.forward.prox(self.transform.inverse(v-u))

            v = self.prior.prox(self.transform(x)+u)

            u = u + self.transform(x) - v

        if return_value == 'both':
            return x, v
        elif return_value == 'x':
            return x
        return v
