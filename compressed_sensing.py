
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import argparse

import tools
import prox
import pnp

parser = argparse.ArgumentParser()
parser.add_argument('--image', type=str, required=True)
parser.add_argument('--sample', type=float, default=0.5)
parser.add_argument('--lambd', type=float, default=0.5)
args = parser.parse_args()

img = Image.open(args.image).convert('L')
img = np.array(img) / 255.

k = int(img.size * args.sample)
ri = np.random.choice(img.size, k, replace=False)
mask = np.zeros(img.shape, dtype=bool)
mask.T.flat[ri] = True
y = img.copy()
y[~mask] = 0.


class DCTMap:
    def __call__(self, x):
        return tools.dct2d(x)

    def inverse(self, v):
        return tools.idct2d(v)

mseloss = prox.MSE_CS(y, mask)
sparse_loss = prox.L1Norm(args.lambd)
optimizer = pnp.PnP_ADMM(mseloss, sparse_loss, y.shape, DCTMap())
recon = optimizer.run(iter=100, return_value='x')

y[~mask] = 1.

plt.subplot(1, 3, 1)
plt.imshow(tools.image_reformat(img), cmap='gray')
plt.subplot(1, 3, 2)
plt.imshow(tools.image_reformat(y), cmap='gray')
plt.subplot(1, 3, 3)
plt.imshow(tools.image_reformat(recon), cmap='gray')
plt.show()
