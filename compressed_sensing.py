
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import argparse

import tools
import prox
import pnp

parser = argparse.ArgumentParser()
parser.add_argument('--image', type=str, required=True)
parser.add_argument('--sample', type=float, default=0.2)
parser.add_argument('--lambd', type=float, default=1e-2)
parser.add_argument('--iter', type=int, default=100)
parser.add_argument('--prior', type=str, default='dct')
args = parser.parse_args()

img = Image.open(args.image).convert('L')
img = np.array(img) / 255.

k = int(img.size * args.sample)
ri = np.random.choice(img.size, k, replace=False)
mask = np.ones(img.shape, dtype=bool) * False
mask.T.flat[ri] = True
y = img.copy()
y[~mask] = 0.


mseloss = prox.MSEwithMask(y, mask)
if args.prior == 'dct':

    sparse_prior = prox.L1Norm(args.lambd)
    class DCT_Transform:
        def __call__(self, x):
            return tools.dct2d(x)

        def inverse(self, v):
            return tools.idct2d(v)
    dct_transform = DCT_Transform()
    optimizer = pnp.PnP_ADMM(mseloss, sparse_prior, img.shape, transform=dct_transform)
    recon = optimizer.run(alpha=0.001, iter=args.iter, return_value='x')

elif args.prior == 'dncnn':

    dncnn_prior = prox.DnCNN_Prior('DnCNN/dncnn_50.pth')
    optimizer = pnp.PnP_ADMM(mseloss, dncnn_prior, img.shape)
    recon = optimizer.run(alpha=0.01, iter=args.iter, return_value='x')


mse = tools.compute_mse(img, recon, reformat=True)
ssim = tools.compute_ssim(img, recon, reformat=True)
print('MSE: {:.5f}'.format(mse))
print('SSIM: {:.5f}'.format(ssim))

tools.stackview([img, y, recon], width=20)
