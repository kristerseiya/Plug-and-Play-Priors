
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
parser.add_argument('--trials', help='number of trials', type=int, default=1)
# parser.add_argument('--lambd', type=float, default=0.05)
parser.add_argument('--noise_lvl', type=float, default=25)
parser.add_argument('--weights', type=str, default='DnCNN/dncnn50.pth')
args = parser.parse_args()

img = Image.open(args.image).convert('L')
img = np.array(img)

mse_result = np.zeros([args.trials])
ssim_result = np.zeros([args.trials])
msssim_result = np.zeros([args.trials])

for i in range(args.trials):
    net = load_dncnn(args.weights)

    noisy = noise.add_gauss(img / 255., args.noise_lvl / 255.)
    x = torch.tensor(noisy, dtype=torch.float32, device=net.device)
    x = x.view(1, 1, *x.size())
    y = net(x)
    y = y.view(y.size(-2), y.size(-1))
    y = y.cpu().numpy()

    noisy = tools.image2uint8(noisy)
    recon = tools.image2uint8(y)

    # reconstruction quality assessment
    mse_result[i] = tools.compute_mse(img, recon)[0]
    ssim_result[i] = tools.ssim(img, recon).mean()
    msssim_result[i] = tools.msssim(img, recon).mean()

print('MSE: {:.5f}'.format(mse_result.mean()))
print('SSIM: {:.5f}'.format(ssim_result.mean()))
print('MS-SSIM: {:.5f}'.format(msssim.mean()))

# noisy_img = Image.fromarray(tools.image_reformat(recon), 'L')
# noisy_img.show()
