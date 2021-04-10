
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import torch
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
parser.add_argument('--trials', help='number of trials', type=int, default=1)
parser.add_argument('--sample', help='sample rate', type=float, default=0.2)
parser.add_argument('--noise', help='gaussian noise level', type=float, default=0)
parser.add_argument('--prior', help='image prior option [\'dct\' or \'dncnn\' or \'tv\' or \'bm3d\']', type=str, default='dncnn')
parser.add_argument('--iter', help='number of iteration', type=int, default=100)
parser.add_argument('--alpha', help='coeefficient of forward model', type=float, default=100.)
parser.add_argument('--lambd', help='coeefficient of prior', type=float, default=1e-2)
parser.add_argument('--weights', help='path to weights', type=str, default='DnCNN/dncnn50.pth')
parser.add_argument('--relax', help='relaxation for ADMM', type=float, default=0.)
args = parser.parse_args()

# read image
img = Image.open(args.image).convert('L')
img = np.array(img) / 255.

mse_result = np.zeros([args.trials])
ssim_result = np.zeros([args.trials])
msssim_result = np.zeros([args.trials])

for i in range(args.trials):
    # do random sampling from the image
    k = int(img.size * args.sample)
    ri = np.random.choice(img.size, k, replace=False)
    mask = np.ones(img.shape, dtype=bool) * False
    mask.T.flat[ri] = True
    y = img.copy()
    if args.noise != 0:
        y = noise.add_gauss(y, std=args.noise / 255.)
    y[~mask] = 0.

    # forward model
    mseloss = prox.MaskMSE(y, mask, alpha=args.alpha)

    # use sparsity in DCT domain as a prior
    if args.prior == 'dct':

        # prior
        sparse_prior = prox.L1Norm(args.lambd)

        # define transformation from x to v
        class DCT_Transform:
            def __call__(self, x):
                return tools.dct2d(x)

            def inverse(self, v):
                return tools.idct2d(v)

        # optimize
        dct_transform = DCT_Transform()
        optimizer = pnp.PnP_ADMM(mseloss, sparse_prior, transform=dct_transform)
        optimizer.init(np.random.rand(*y.shape), np.zeros_like(y))
        recon = optimizer.run(iter=args.iter, relax=args.relax, return_value='x')

    # use trained prior from DnCNN
    elif args.prior == 'dncnn':

        dncnn_prior = prox.DnCNN_Prior(args.weights, use_tensor=True)
        y_t = torch.tensor(y, dtype=torch.float32, requires_grad=False, device=dncnn_prior.device)
        y_t = y_t.view(1, 1, *y_t.size())
        mask_t = torch.tensor(mask, dtype=bool, requires_grad=False, device=dncnn_prior.device)
        mask_t = mask_t.view(1, 1, *mask_t.size())
        mseloss = prox.MaskMSETensor(y_t, mask_t, args.alpha)
        optimizer = pnp.PnP_ADMM(mseloss, dncnn_prior)
        optimizer.init(torch.rand_like(y_t), torch.zeros_like(y_t))
        recon_t = optimizer.run(iter=args.iter, relax=args.relax, return_value='x')
        recon = recon_t.cpu().numpy().squeeze(0).squeeze(0)

    # total variation norm
    elif args.prior == 'tv':

        tv_prior = prox.TVNorm(args.lambd)
        optimizer = pnp.PnP_ADMM(mseloss, tv_prior)
        optimizer.init(np.random.rand(*y.shape), np.zeros_like(y))
        recon = optimizer.run(iter=args.iter, relax=args.relax, return_value='x')

    # block matching with 3D filter
    elif args.prior == 'bm3d':

        bm3d_prior = prox.BM3D_Prior(args.lambd)

        optimizer = pnp.PnP_ADMM(mseloss, bm3d_prior)
        optimizer.init(np.random.rand(*y.shape), np.zeros_like(y))
        recon = optimizer.run(iter=args.iter, relax=args.relax, return_value='x')


    # reconstruction quality assessment
    mse_result[i] = tools.compute_mse(img, recon, scale=1)[0]
    ssim_result[i] = tools.ssim(img, recon, scale=1).mean()
    msssim_result[i] = tools.msssim(img, recon, scale=1).mean()

print('MSE: {:.5f}'.format(mse_result.mean()))
print('SSIM: {:.5f}'.format(ssim_result.mean()))
print('MS-SSIM: {:.5f}'.format(msssim_result.mean()))


# # viewer
# tools.stackview([img, y, recon], width=20, method='Pillow')

# # save result
# if args.save != None:
#     if not os.path.exists(args.save):
#         os.makedirs(args.save)
#     image_name = args.image.split('/')[-1]
#     image_name = image_name.split('.')[-2]
#     key = datetime.now().strftime('%m%d%H%M%S')
#     rate = str(args.sample).replace('.', '')
#     original_name = image_name + '_orignal.png'
#     compressed_name = image_name + '_compressed_' + rate + '_' + key + '.png'
#     restored_name = image_name + '_restored_' + rate + '_' + args.prior + '_' + key + '.png'
#     original_path = os.path.join(args.save, original_name)
#     compressed_path = os.path.join(args.save, compressed_name)
#     restored_path = os.path.join(args.save, restored_name)
#     Image.fromarray(tools.image2uint8(img), 'L').save(original_path)
#     Image.fromarray(tools.image2uint8(y), 'L').save(compressed_path)
#     Image.fromarray(tools.image2uint8(recon), 'L').save(restored_path)
