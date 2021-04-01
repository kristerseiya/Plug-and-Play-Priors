
import argparse
import torch
from torch.utils.data import DataLoader
from torch.optim import Adam
import numpy as np
from PIL import Image

import DnCNN
import tools

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
    testldr = DataLoader(testset, batch_size=64, shuffle=False)

    net = model.DnCNN().move(device)
    optimizer = Adam(net.parameters(), lr=1e-4)

    log = run.train(net, optimizer, args.n_epoch, trainldr, args.noise_lvl, validation=valldr)

    torch.save(net.state_dict(), args.save)

elif args.command == 'run':

    image = Image.open(args.image).convert('L')
    x = np.array(image)
    x = x + np.random.normal(size=x.shape) * args.noise_lvl
    x = np.clip(x, 0, 255)
    noisy = Image.fromarray(x.astype(np.uint8), 'L')
    net = DnCNN.DnCNN(act_mode='BR')
    net.load_state_dict(torch.load(args.weights, map_location=device))
    recon = DnCNN.inference(net, noisy)

    _, psnr = tools.compute_mse(np.array(image), np.array(recon), reformat=False)
    print('PSNR: {:.3}dB'.format(psnr))
    tools.stackview([np.array(image) / 255., np.array(noisy) / 255., np.array(recon) / 255.])
