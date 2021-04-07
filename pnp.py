
import numpy as np
import copy


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

    def init(self, v, u=None):
        self.v_init = copy.deepcopy(v)
        if u == None:
            self.u_init = copy.deepcopy(v)
        else:
            self.u_init = copy.deepcopy(u)

    def run(self, iter=100, relax=0., return_value='both', verbose=True):

        # v = np.zeros(self.prior.input_shape)
        # u = np.zeros(self.prior.input_shape)
        v = self.v_init
        u = self.u_init

        for i in range(iter):

            if verbose:
                print('Iteration #{:d}'.format(i+1))

            x = self.forward(self.transform.inverse(v-u))

            x_relaxed = (1 - relax) * self.transform(x) + relax * v
            v = self.prior(x_relaxed+u)

            diff = x_relaxed - v
            u = u + diff

        if return_value == 'both':
            return x, v
        elif return_value == 'x':
            return x
        return v
