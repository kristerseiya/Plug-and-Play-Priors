
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from skimage.metrics import structural_similarity as skssim
import argparse

import tools
import noise
import prox
from DnCNN import load_dncnn
import torch

parser = argparse.ArgumentParser()
parser.add_argument('--image', type=str, required=True)
parser.add_argument('--lambd', type=float, default=0.05)
parser.add_argument('--noise_lvl', type=float, default=25)
parser.add_argument('--weights', type=str, default='DnCNN/dncnn50.pth')
args = parser.parse_args()

img = Image.open(args.image).convert('L')
img = np.array(img)
noisy = noise.add_gauss(img / 255., args.noise_lvl / 255.)
#noisy = noise.add_poisson(img, 70.)

net = load_dncnn(args.weights)
x = torch.tensor(noisy, dtype=torch.float32, device=net.device)
x = x.view(1, 1, *x.size())
y = net(x)
y = y.view(y.size(-2), y.size(-1))
y = y.cpu().numpy()
# tvnorm = prox.TVNorm(args.lambd)
# tvnorm.set(1.)
# recon = tvnorm.prox(noisy)

noisy = tools.image2uint8(noisy)
recon = tools.image2uint8(y)

mse, psnr = tools.compute_mse(img, recon)
ssim = tools.ssim(img, recon).mean()
msssim = tools.msssim(img, recon).mean()
# ssim2 = skssim(img, recon, data_range=255)
print('MSE: {:.5f}'.format(mse))
print('PSNR: {:.5f}'.format(psnr))
print('SSIM: {:.5f}'.format(ssim))
print('MS-SSIM: {:.5f}'.format(msssim))

tools.stackview([img, noisy, recon], width=20, method='Pillow')
# noisy_img = Image.fromarray(tools.image_reformat(recon), 'L')
# noisy_img.show()
