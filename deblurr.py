
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import argparse
from scipy.signal import correlate2d

import tools
import prox
import pnp
import noise

# command line arguments
parser = argparse.ArgumentParser()
parser.add_argument('--image', help='path to image', type=str, required=True)
parser.add_argument('--window', help='window size', type=int, default=15)
parser.add_argument('--sigma', help='gaussian deviation', type=float, default=1.5)
parser.add_argument('--prior', help='image prior option [\'dct\' or \'dncnn\' or \'tv\' or \'bm3d\']', type=str, default='dncnn')
parser.add_argument('--iter', help='number of iteration', type=int, default=100)
parser.add_argument('--alpha', help='coeefficient of forward model', type=float, default=1000)
parser.add_argument('--lambd', help='coeefficient of prior', type=float, default=1e-2)
parser.add_argument('--weights', help='path to weights', type=str, default='DnCNN/dncnn50.pth')
parser.add_argument('--save', help='a directory to save result', type=str, default=None)
parser.add_argument('--relax', help='relaxation for ADMM', type=float, default=0.)
parser.add_argument('--verbose', action='store_true')
args = parser.parse_args()

# read image
img = Image.open(args.image).convert('L')
img = np.array(img) / 255.
gauss_window = tools.get_gauss2d(args.window, args.window, args.sigma)
y = correlate2d(img, gauss_window, mode='same', boundary='wrap')

# forward model
forward = prox.CirculantMSE(y, gauss_window, input_shape=img.shape, alpha=args.alpha)

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
    optimizer = pnp.PnP_ADMM(forward, sparse_prior, transform=dct_transform)
    optimizer.init(np.zeros_like(y))
    recon = optimizer.run(iter=args.iter, relax=args.relax, return_value='x', verbose=args.verbose)

# use trained prior from DnCNN
elif args.prior == 'dncnn':

    dncnn_prior = prox.DnCNN_Prior(args.weights, use_tensor=False)
    optimizer = pnp.PnP_ADMM(forward, dncnn_prior)
    optimizer.init(np.zeros_like(y))
    recon = optimizer.run(iter=args.iter, relax=args.relax, return_value='x', verbose=args.verbose)

# total variation norm
elif args.prior == 'tv':

    tv_prior = prox.TVNorm(args.lambd)
    optimizer = pnp.PnP_ADMM(forward, tv_prior)
    optimizer.init(np.zeros_like(y))
    recon = optimizer.run(iter=args.iter, relax=args.relax, return_value='x', verbose=args.verbose)

# block matching with 3D filter
elif args.prior == 'bm3d':

    bm3d_prior = prox.BM3D_Prior(args.lambd)
    optimizer = pnp.PnP_ADMM(forward, bm3d_prior)
    optimizer.init(np.zeros_like(y))
    recon = optimizer.run(iter=args.iter, relax=args.relax, return_value='x', verbose=args.verbose)


# reconstruction quality assessment
mse, psnr = tools.compute_mse(img, recon, scale=255)
ssim = tools.ssim(img, recon, scale=255).mean()
print('MSE: {:.5f}'.format(mse))
print('PSNR: {:.5f}'.format(psnr))
print('SSIM: {:.5f}'.format(ssim))

# viewer
tools.stackview([img, y, recon], width=20, method='Pillow')

if args.save != None:
    if not os.path.exists(args.save):
        os.makedirs(args.save)
    image_name = args.image.split('/')[-1]
    image_name = image_name.split('.')[-2]
    key = datetime.now().strftime('%m%d%H%M%S')
    w_size = str(args.window)
    sig = str(args.sigma).replace('.', '')
    original_name = image_name + '_orignal.png'
    blurred_name = image_name + '_blurred_' + w_size + '_' + sig + '_' + key + '.png'
    restored_name = image_name + '_restored_' + w_size + '_' + sig + '_' + args.prior + '_' + key + '.png'
    original_path = os.path.join(args.save, original_name)
    blurred_name = os.path.join(args.save, blurred_name)
    restored_path = os.path.join(args.save, restored_name)
    Image.fromarray(tools.image2uint8(img), 'L').save(original_path)
    Image.fromarray(tools.image2uint8(y), 'L').save(blurred_name)
    Image.fromarray(tools.image2uint8(recon), 'L').save(restored_path)
