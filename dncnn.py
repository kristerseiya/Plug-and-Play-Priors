
import argparse
import torch
from torch.utils.data import DataLoader
from torch.optim import Adam
import numpy as np
from PIL import Image

import DnCNN
import tools
import prox

parser = argparse.ArgumentParser()
parser.add_argument('--command', type=str, default='run')
parser.add_argument('--weights', type=str, default=None)
parser.add_argument('--image', type=str, default=None)
parser.add_argument('--noise_lvl', type=float, default=30.)
parser.add_argument('--data_dir', type=str, default=None)
parser.add_argument('--n_epoch', type=int, default=20)
parser.add_argument('--save', type=str, default='result.pth')

args = parser.parse_args()


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

if args.command == 'train':

    dataset = data.ImageDataset(data_dirs, mode='none', store='disk', repeat=10)
    trainset, valset, testset = dataset.split(0.7, 0.1, 0.2)

    trainldr = DataLoader(trainset, batch_size=64, shuffle=True, drop_last=True)
    valldr = DataLoader(valset, batch_size=64, shuffle=False)
    testldr = DataLoader(testset, batch_size=1, shuffle=False)

    net = model.DnCNN().move(device)
    optimizer = Adam(net.parameters(), lr=1e-4)

    log = run.train(net, optimizer, args.n_epoch, trainldr, args.noise_lvl, validation=valldr)

    torch.save(net.state_dict(), args.save)

elif args.command == 'run':

    img = Image.open(args.image).convert('L')
    img = np.array(img) / 255.
    noisy = img + np.random.normal(size=img.shape) * args.noise_lvl / 255.

    net = prox.DnCNN_Prior(args.weights, None)
    recon = net(noisy)
    # net = DnCNN.DnCNN(act_mode='BR')
    # net.load_state_dict(torch.load(args.weights, map_location=device))
    # recon = DnCNN.inference(net, noisy)

    _, psnr = tools.compute_mse(tools.image2uint8(img), tools.image2uint8(recon))
    print('PSNR: {:.3}dB'.format(psnr))
    tools.stackview([img, noisy, recon], method='Pillow')
