
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import torch
import argparse
import os
from datetime import datetime

import utils
import func
import pnp
import noise

from Denoisers import dnsr

# command line arguments
parser = argparse.ArgumentParser()
parser.add_argument('--image', help='path to image', type=str, required=True)
parser.add_argument('--idx', help='File with index of sampled points', type=str, default=None)
parser.add_argument('--sample', help='sample rate', type=float, default=0.2)
parser.add_argument('--noise', help='gaussian noise level', type=float, default=0)
parser.add_argument('--prior', help='image prior option [\'dct\' or \'dncnn\' or \'tv\' or \'bm3d\']', type=str, default='dncnn')
parser.add_argument('--iter', help='number of iteration', type=int, default=100)
parser.add_argument('--alpha', help='coefficient of forward model', type=float, default=100.)
parser.add_argument('--lambd', help='coefficient of prior', type=float, default=1e-2)
parser.add_argument('--sigma', help='bm3d parameter', type=float, default=3)
parser.add_argument('--weights', help='path to weights', type=str, default='DnCNN/dncnn50.pth')
parser.add_argument('--save_recon', help='file name for recoonstructed image', type=str, default=None)
parser.add_argument('--save_idx', help='file name for storing index', type=str, default=None)
parser.add_argument('--relax', help='relaxation for ADMM', type=float, default=0.)
parser.add_argument('--verbose', action='store_true')
args = parser.parse_args()

# read image
img = Image.open(args.image).convert('L')
img = np.array(img) / 255.

if args.idx != None:
    ri = np.fromfile(args.idx, dtype=np.int32)
else:
    # do random sampling from the image
    k = int(img.size * args.sample)
    ri = np.random.choice(img.size, k, replace=False)

print('{:d}/{:d} pixels'.format(len(ri), img.size))

mask = np.ones(img.shape, dtype=bool) * False
mask.T.flat[ri] = True
y = img.copy()
if args.noise != 0:
    y = noise.add_gauss(y, std=args.noise / 255.)
y[~mask] = 0.

# forward model
mseloss = func.MaskMSE(y, mask, alpha=args.alpha)

# use sparsity in DCT domain as a prior
if args.prior == 'dct':

    class SparseDCT:
        def __init__(self, lambd):
            self.lambd = lambd

        def __call__(self, x):
            fx = utils.dct2d(x)
            fx = np.maximum(np.abs(fx) - self.lambd, 0) * np.sign(fx)
            return utils.idct2d(fx)

    # prior
    sparse_dct = SparseDCT(args.lambd)

    # optimize
    optimizer = pnp.PnPADMM(mseloss, sparse_dct)
    optimizer.init(np.random.rand(*y.shape), np.zeros_like(y))
    recon = optimizer.run(iter=args.iter, relax=args.relax, return_value='x', verbose=args.verbose)

# use trained prior from DnCNN
elif args.prior in ['dncnn', 'cdncnn']:

    if args.prior == 'dncnn':
        dncnn = dnsr.DnCNN(args.weights, use_tensor=True)
    else:
        dncnn = dnsr.cDnCNN(args.weights, use_tensor=True)
        dncnn.set_param(args.sigma / 255.)
    y_t = torch.tensor(y, dtype=torch.float32, requires_grad=False, device=dncnn.device)
    y_t = y_t.view(1, 1, *y_t.size())
    mask_t = torch.tensor(mask, dtype=bool, requires_grad=False, device=dncnn.device)
    mask_t = mask_t.view(1, 1, *mask_t.size())
    mseloss = func.MaskMSETensor(y_t, mask_t, args.alpha)
    optimizer = pnp.PnPADMM(mseloss, dncnn)
    optimizer.init(torch.rand_like(y_t), torch.zeros_like(y_t))
    recon_t = optimizer.run(iter=args.iter,
                            relax=args.relax,
                            return_value='x',
                            verbose=args.verbose)
    recon = recon_t.cpu().numpy().squeeze(0).squeeze(0)

# total variation norm
elif args.prior == 'tv':

    tvprox = dnsr.ProxTV(args.lambd)
    optimizer = pnp.PnPADMM(mseloss, tvprox)
    optimizer.init(np.random.rand(*y.shape), np.zeros_like(y))
    recon = optimizer.run(iter=args.iter, relax=args.relax, return_value='x', verbose=args.verbose)

# block matching with 3D filter
elif args.prior == 'bm3d':

    bm3d = dnsr.BM3D()
    bm3d.set_param(args.sigma / 255.)

    optimizer = pnp.PnPADMM(mseloss, bm3d)
    optimizer.init(np.random.rand(*y.shape), np.zeros_like(y))
    recon = optimizer.run(iter=args.iter, relax=args.relax, return_value='x', verbose=args.verbose)

# reconstruction quality assessment
mse, psnr = utils.compute_mse(img, recon, scale=1)
ssim = utils.ssim(img, recon, scale=1).mean()
msssim = utils.msssim(img, recon, scale=1).mean()
print('MSE: {:.5f}'.format(mse))
print('PSNR: {:.5f}'.format(psnr))
print('SSIM: {:.5f}'.format(ssim))
print('MS-SSIM: {:.5f}'.format(msssim))

# viewer
utils.stackview([img, y, recon], width=20, method='Pillow')

# save result
if args.save_recon != None:
    Image.fromarray(utils.image2uint8(recon), 'L').save(args.save_recon + '.png')

if args.idx == None and args.save_idx != None:
    ri.astype(np.int32).tofile(args.save_idx)
