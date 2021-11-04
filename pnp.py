
import numpy as np
import copy


class PnPADMM:
    def __init__(self, forward, denoiser):
        self.forward = forward
        self.denoiser = denoiser

    def init(self, v, u):
        self.v_init = copy.deepcopy(v)
        self.u_init = copy.deepcopy(u)

    def run(self, iter=100, relax=0., return_value='both', verbose=False):

        v = self.v_init
        u = self.u_init

        for i in range(iter):

            if verbose:
                print('Iteration #{:d}'.format(i+1))

            x = self.forward.prox(v-u)

            x_relaxed = (1 - relax) * x + relax * v
            v = self.denoiser(x_relaxed+u)

            diff = x_relaxed - v
            u = u + diff

        if return_value == 'both':
            return x, v
        elif return_value == 'x':
            return x
        return v
