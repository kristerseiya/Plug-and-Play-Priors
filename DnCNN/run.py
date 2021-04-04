
import torch
import torch.nn.functional as F
import numpy as np
import torch
import os
from PIL import Image
from tqdm import tqdm
import sys

from . import data

def train_single_epoch(model, optimizer, train_loader,
                       noise_lvl, clip=False, lossfn='L2',
                       scheduler=None):

    n_data = 0
    sigma = noise_lvl

    if lossfn.upper() == 'L2':
        lossfn = F.mse_loss
    elif lossfn.upper() == 'L1':
        lossfn = F.l1_loss

    total_loss = 0.
    model.train()

    pbar = tqdm(total=len(train_loader), position=0, leave=False, file=sys.stdout)

    for images in train_loader:
        optimizer.zero_grad()
        batch_size = images.size(0)
        images = images.to(model.device)
        if type(noise_lvl) == list:
            sigma = torch.rand(batch_size, 1, 1, 1, device=model.device)
            sigma = sigma * (noise_lvl[1] - noise_lvl[0]) + noise_lvl[0]
        noise = torch.randn_like(images) * sigma / 255.
        noisy = images + noise
        if clip:
            noisy = torch.clip(noisy, 0, 1)
        output = model(noisy)
        loss = lossfn(output, images)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * batch_size
        n_data += batch_size
        if scheduler != None:
            scheduler.step()
        pbar.update(1)

    tqdm.close(pbar)

    return total_loss / float(n_data)

@torch.no_grad()
def validate(model, test_loader, noise_lvl, clip=False, lossfn='L2'):

    n_data = 0
    sigma = noise_lvl

    if lossfn.upper() == 'L2':
        lossfn = F.mse_loss
    elif lossfn.upper() == 'L1':
        lossfn = F.l1_loss

    total_loss = 0.
    model.eval()

    pbar = tqdm(total=len(test_loader), position=0, leave=False, file=sys.stdout)

    for images in test_loader:
        batch_size = images.size(0)
        images = images.to(model.device)
        if type(noise_lvl) == list:
            sigma = torch.rand(batch_size, 1, 1, 1, device=model.device)
            sigma = sigma * (noise_lvl[1] - noise_lvl[0]) + noise_lvl[0]
        noise = torch.randn_like(images) * sigma / 255.
        noisy = images + noise
        if clip:
            noisy = torch.clip(noisy, 0, 1)
        output = model(noisy)
        total_loss += lossfn(output, images).item() * batch_size
        n_data += batch_size
        pbar.update(1)

    tqdm.close(pbar)

    return total_loss / float(n_data)


def train(model, optimizer, max_epoch, train_loader, noise_lvl, clip=False, lossfn='L2',
          validation=None, scheduler=None, lr_step='epoch',
          checkpoint_dir=None, max_tolerance=-1):

    best_loss = 99999.
    tolerated = 0

    if lr_step == 'epoch':
        lr_step_per_epoch = True
    elif lr_step == 'batch':
        lr_step_per_epoch = False
    else:
        lr_step_per_epoch = True

    _scheduler = None
    if not lr_step_per_epoch:
        _scheduler = scheduler

    log = np.zeros([max_epoch, 2], dtype=np.float)

    for e in range(max_epoch):

        print('\nEpoch #{:d}'.format(e+1))

        log[e, 0] = train_single_epoch(model, optimizer, train_loader, noise_lvl,
                                       clip, lossfn, _scheduler)

        print('Train Loss: {:.5f}'.format(log[e, 0]))

        if scheduler != None and lr_step_per_epoch:
            scheduler.step()

        if validation != None:

            log[e, 1] = validate(model, validation, noise_lvl, clip, lossfn)

            print('Val Loss: {:.5f}'.format(log[e, 1]))

            if (checkpoint_dir != None) and (best_loss > log[e, 1]):
                best_loss = log[e, 1]
                if not os.path.exists(checkpoint_dir):
                    os.makedirs(checkpoint_dir)
                checkpoint_path = os.path.join(checkpoint_dir, 'checkpoint'+str(e+1)+'.pth')
                torch.save(model.state_dict(), checkpoint_path)
                print('Best Loss! Saved.')
            elif max_tolerance >= 0:
                tolerated += 1
                if tolerated > max_tolerance:
                    return log[0:e, :]

    return log

@torch.no_grad()
def inference(model, image):
    model.eval()
    transform = data.get_transform('test')
    x = transform(image)
    x = x.unsqueeze(0)
    x = x.to(model.device)
    x = model(x)
    x = x.squeeze(0)
    x = x.cpu().numpy()
    x = x.transpose([1, 2, 0])
    x = x.squeeze(-1)
    x = np.clip(x, 0, 1) * 255
    return Image.fromarray(x.astype(np.uint8), 'L')
