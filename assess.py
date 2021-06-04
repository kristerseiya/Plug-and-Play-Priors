
from PIL import Image
import numpy as np
import argparse

import tools

parser = argparse.ArgumentParser()
parser.add_argument('--image1', type=str, required=True)
parser.add_argument('--image2', type=str, required=True)
args = parser.parse_args()

img1 = np.array(Image.open(args.image1).convert('L'))
img2 = np.array(Image.open(args.image2).convert('L'))

mse, psnr = tools.compute_mse(img1, img2, scale=255)
ssim = tools.ssim(img1, img2, scale=255).mean()
msssim = tools.msssim(img1, img2, scale=255).mean()
print('MSE: {:.5f}'.format(mse))
print('PSNR: {:.5f}'.format(psnr))
print('SSIM: {:.5f}'.format(ssim))
print('MS-SSIM: {:.5f}'.format(msssim))
