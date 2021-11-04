
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import argparse
from scipy.signal import correlate2d

import utils
import func
import pnp
import noise

from Denoisers import dnsr

# command line arguments
parser = argparse.ArgumentParser()
parser.add_argument('--image', help='path to image', type=str, required=True)
parser.add_argument('--window', help='window size', type=int, default=15)
parser.add_argument('--sigma', help='gaussian deviation', type=float, default=1.5)
parser.add_argument('--prior', help='image prior option [\'dct\' or \'dncnn\' or \'tv\' or \'bm3d\']', type=str, default='dncnn')
parser.add_argument('--iter', help='number of iteration', type=int, default=10)
parser.add_argument('--alpha', help='coeefficient of forward model', type=float, default=1000)
parser.add_argument('--lambd', help='coeefficient of prior', type=float, default=1e-2)
parser.add_argument('--sigma', help='bm3d parameter', type=float, default=3)
parser.add_argument('--weights', help='path to weights', type=str, default='DnCNN/dncnn50.pth')
parser.add_argument('--save', help='a directory to save result', type=str, default=None)
parser.add_argument('--relax', help='relaxation for ADMM', type=float, default=0.)
parser.add_argument('--verbose', action='store_true')
args = parser.parse_args()

# read image
img = Image.open(args.image).convert('L')
img = np.array(img) / 255.
gauss_window = utils.get_gauss2d(args.window, args.window, args.sigma)
y = correlate2d(img, gauss_window, mode='same', boundary='wrap')

# forward model
forward = func.CirculantMSE(y, gauss_window, input_shape=img.shape, alpha=args.alpha)

# use sparsity in DCT domain as a prior
if args.prior == 'dct':

    class SparseDCT:
        def __init__(self, lambd):
            self.lambd = lambd

        def __call__(self, x):
            fx = utils.dct2d(x)
            fx = np.maximum(np.abs(fx) - self.lambd, 0) * np.sign(fx)
            return utils.idct2d(fx)

    # optimize
    sparse_dct = DCT_Transform()
    optimizer = pnp.PnPADMM(forward, sparse_dct)
    optimizer.init(np.random.rand(*y.shape), np.zeros_like(y))
    recon = optimizer.run(iter=args.iter, relax=args.relax, return_value='x', verbose=args.verbose)

# use trained prior from DnCNN
elif args.prior in ['dncnn', 'cdncnn']:

    if args.prior == 'dncnn':
        dncnn = dnsr.DnCNN(args.weights, use_tensor=True)
    else:
        dncnn = dnsr.cDnCNN(args.weights, use_tensor=True)
        dncnn.set_param(args.sigma / 255.)
    optimizer = pnp.PnPADMM(forward, dncnn)
    optimizer.init(np.random.rand(*y.shape), np.zeros_like(y))
    recon = optimizer.run(iter=args.iter, relax=args.relax, return_value='x', verbose=args.verbose)

# total variation norm
elif args.prior == 'tv':

    tvprox = dnsr.TVNorm(args.lambd)
    optimizer = pnp.PnPADMM(forward, tvprox)
    optimizer.init(np.random.rand(*y.shape), np.zeros_like(y))
    recon = optimizer.run(iter=args.iter, relax=args.relax, return_value='x', verbose=args.verbose)

# block matching with 3D filter
elif args.prior == 'bm3d':

    bm3d = dnsr.BM3D()
    bm3d.set_param(args.sigma / 255.)
    optimizer = pnp.PnPADMM(forward, bm3d)
    optimizer.init(np.random.rand(*y.shape), np.zeros_like(y))
    recon = optimizer.run(iter=args.iter, relax=args.relax, return_value='x', verbose=args.verbose)

img = utils.image2uint8(img)
recon = utils.image2uint8(recon)

# reconstruction quality assessment
mse, psnr = utils.compute_mse(img, recon, scale=255)
ssim = utils.ssim(img, recon, scale=255).mean()
print('MSE: {:.5f}'.format(mse))
print('PSNR: {:.5f}'.format(psnr))
print('SSIM: {:.5f}'.format(ssim))

# viewer
utils.stackview([img, y, recon], width=20, method='Pillow')

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
    Image.fromarray(utils.image2uint8(img), 'L').save(original_path)
    Image.fromarray(utils.image2uint8(y), 'L').save(blurred_name)
    Image.fromarray(utils.image2uint8(recon), 'L').save(restored_path)
