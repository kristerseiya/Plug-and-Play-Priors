
import argparse
import glob
import os
import numpy as np

import noise
import prox
import pnp
import tools

parser = argparse.ArgumentParser()
parser.add_argument('--weights', help='path to weights', type=str, required=True)
parser.add_argument('--test_images', help='path to image', type=str, required=True)
parser.add_argument('--trials', help='sample rate', type=int, default=20)
parser.add_argument('--iter', help='number of iteration', type=int, default=100)
parser.add_argument('--alphas', type=str, default='1:2000:20')
# parser.add_argument('--task', type=str, default='denoise')
parser.add_argument('--noise', help='gaussian noise level', type=float, default=0)
parser.add_argument('--save', help='file name to save results', type=str, default='results.txt')
parser.add_argument('--relax', help='relaxation for ADMM', type=float, default=0.)
args = parser.parse_args()

prior = prox.DnCNN_Prior(args.weights, input_shape=None)

alphas = args.alphas.split(':')
assert(len(alphas)==2 or len(alphas)==3)
if len(alphas)==2:
    n_points = 20
else:
    n_points = int(alphas[2])

alphas = np.logspace(start=int(alphas[0]), stop=int(alphas[1]), num=n_points)

img_path_list = list()
for ext in ['png', 'jpg']:
    img_path_list += list(glob.glob(os.path.join(args.test_images, '*.'+ext)))


samp_rates = np.linspace(0.1, 0.8. 0.05)
log = np.zeros(len(img_path_list), len(samp_rates))

for i, img_path in enumerate(img_path_list):

    img = np.array(Image.open(img_path).convert('L')) / 255.

    prior.input_shape = img.shape

    for s, samp_rate in enumerate(samp_rates):

        trial_exp = np.zeros([trials])

        for t in range(trials):

            k = int(img.size * args.sample)
            ri = np.random.choice(img.size, k, replace=False)
            mask = np.ones(img.shape, dtype=bool) * False
            mask.T.flat[ri] = True
            y = img.copy()
            if args.noise != 0:
                y = noise.add_gauss(y, std=args.noise/255.)
            y[~mask] = 0.

            alpha_exp = np.zeros([len(alphas), ])

            for a, alpha in enumerate(alphas):

                forward = prox.MSEwithMask(y, mask, alpha, input_shape=img.shape)
                optimizer = pnp.PnP_ADMM(forward, prior)
                recon = optimizer.run(iter=args.iter, 0, return_value='x', verbose=False)
                alpha_exp[a] = tools.compute_mse(image2uint8(img), image2uint8(recon))[0]

            trial_exp[t] = np.min(alpha_exp)

        log[i, s] = trial_exp.mean()


avg_mse = log.mean(axis=-1)

log = 10 * np.log10(255**2 / log)
avg_psnr = 10 * np.log10(255**2 / avg_mse)

result_file = open(args.save, 'w')

result_file.write(args.test_images)

line = ('{:.2f}'*len(samp_rates)).format(*samp_rates)
result_file.write(line)

for l in len(log):
    img_name = img_path_list[l].split('/')[-1]
    line = img_name + ': ' + ('{:.3f}'*len(samp_rates)).format(*log[l, :])
    result_file.write(line)
line = ('{:.3f}'*len(samp_rates)).format(*avg_psnr)
result_file.write(line)
result_file.close()
