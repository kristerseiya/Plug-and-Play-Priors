
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import argparse
import os
from datetime import datetime

import tools
import prox
import pnp
import noise

# command line arguments
parser = argparse.ArgumentParser()
parser.add_argument('--image', help='path to image', type=str, required=True)
parser.add_argument('--sample', help='sample rate', type=float, default=0.2)
parser.add_argument('--lambd', help='coeefficient of prior', type=float, default=1e-2)
parser.add_argument('--noise', help='gaussian noise level', type=float, default=0)
parser.add_argument('--iter', help='number of iteration', type=int, default=100)
parser.add_argument('--prior', help='image prior option [\'dct\' or \'dncnn\' or \'tv\' or \'bm3d\']', type=str, default='dct')
parser.add_argument('--weights', help='path to weights', type=str, default='DnCNN/dncnn50.pth')
parser.add_argument('--alpha', help='coeefficient of forward model', type=float, default=100.)
parser.add_argument('--save', help='a directory to save result', type=str, default=None)
parser.add_argument('--relax', type=float, default=0.)
args = parser.parse_args()

# read image
img = Image.open(args.image).convert('L')
img = np.array(img) / 255.

# do random sampling from the image
k = int(img.size * args.sample)
ri = np.random.choice(img.size, k, replace=False)
mask = np.ones(img.shape, dtype=bool) * False
mask.T.flat[ri] = True
y = img.copy()
if args.noise != 0:
    y = noise.add_gauss(y, std=args.noise)
y[~mask] = 0.

# forward model
mseloss = prox.MSEwithMask(y, mask, alpha=args.alpha, input_shape=img.shape)

# use sparsity in DCT domain as a prior
if args.prior == 'dct':

    # prior
    sparse_prior = prox.L1Norm(args.lambd, input_shape=img.shape)

    # define transformation from x to v
    class DCT_Transform:
        def __call__(self, x):
            return tools.dct2d(x)

        def inverse(self, v):
            return tools.idct2d(v)

    # optimize
    dct_transform = DCT_Transform()
    optimizer = pnp.PnP_ADMM(mseloss, sparse_prior, transform=dct_transform)
    recon = optimizer.run(iter=args.iter, relax=args.relax, return_value='x')

# use trained prior from DnCNN
elif args.prior == 'dncnn':

    dncnn_prior = prox.DnCNN_Prior(args.weights, input_shape=img.shape)
    optimizer = pnp.PnP_ADMM(mseloss, dncnn_prior)
    recon = optimizer.run(iter=args.iter, relax=args.relax, return_value='x')

# total variation norm
elif args.prior == 'tv':

    tv_prior = prox.TVNorm(args.lambd, input_shape=img.shape)
    optimizer = pnp.PnP_ADMM(mseloss, tv_prior)
    recon = optimizer.run(iter=args.iter, relax=args.relax, return_value='x')

# block matching with 3D filter
elif args.prior == 'bm3d':

    bm3d_prior = prox.BM3D_Prior(args.lambd, input_shape=img.shape)

    optimizer = pnp.PnP_ADMM(mseloss, bm3d_prior)
    recon = optimizer.run(iter=args.iter, relax=args.relax, return_value='x')


img = tools.image2uint8(img)
recon = tools.image2uint8(recon)

# reconstruction quality assessment
mse, psnr = tools.compute_mse(img, recon)
ssim = tools.compute_ssim(img, recon)
print('MSE: {:.5f}'.format(mse))
print('PSNR: {:.5f}'.format(psnr))
print('SSIM: {:.5f}'.format(ssim))

# viewer
tools.stackview([img, y, recon], width=20, method='Pillow')

# save result
if args.save != None:
    if not os.path.exists(args.save):
        os.makedirs(args.save)
    image_name = args.image.split('/')[-1]
    image_name = image_name.split('.')[-2]
    key = datetime.now().strftime('%Y%m%d%H%M%S')
    rate = str(args.sample).replace('.', '')
    original_name = image_name + '_orignal.png'
    compressed_name = image_name + '_compressed_' + rate + '_' + key + '.png'
    restored_name = image_name + '_restored_' + rate + '_' + args.prior + '_' + key + '.png'
    original_path = os.path.join(args.save, original_name)
    compressed_path = os.path.join(args.save, compressed_name)
    restored_path = os.path.join(args.save, restored_name)
    Image.fromarray(tools.image2uint8(img), 'L').save(original_path)
    Image.fromarray(tools.image2uint8(y), 'L').save(compressed_path)
    Image.fromarray(tools.image2uint8(recon), 'L').save(restored_path)
