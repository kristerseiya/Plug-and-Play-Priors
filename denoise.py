
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
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
img = np.array(img) / 255.
noisy = noise.add_gauss(img, args.noise_lvl / 255.)
#noisy = noise.add_poisson(img, 70.)

net = load_dncnn(args.weights)
x = torch.tensor(noisy, dtype=torch.float32, device=net.device)
x = x.view(1, 1, *x.size())
y = net(x)
y = y.view(y.size(-2), y.size(-1))
recon = y.cpu().numpy()

mse, psnr = tools.compute_mse(img, recon, 1)
ssim = tools.ssim(img, recon, 1).mean()
msssim = tools.msssim(img, recon, 1).mean()
print('MSE: {:.5f}'.format(mse))
print('PSNR: {:.5f}'.format(psnr))
print('SSIM: {:.5f}'.format(ssim))
print('MS-SSIM: {:.5f}'.format(msssim))

tools.stackview([img, noisy, recon], width=20, method='Pillow')
