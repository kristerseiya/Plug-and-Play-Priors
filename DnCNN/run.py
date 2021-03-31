
import torch
import torch.nn.functional as F
import numpy as np
import torch
import os
from PIL import Image
import tqdm
import sys

from . import data

def train_single_epoch(model, optimizer, train_loader, noise_lvl):

    dataset_size = len(train_loader.dataset)

    total_loss = 0.
    model.train()

    pbar = tqdm.tqdm(total=len(train_loader), position=0, leave=True, file=sys.stdout)

    for images in train_loader:
        optimizer.zero_grad()
        images = images.to(model.device)
        noise = torch.randn_like(images) * noise_lvl / 255.
        noisy = images + noise
        output = model(noisy)
        loss = F.mse_loss(output, images)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * images.size(0)
        pbar.update(1)

    return total_loss / float(dataset_size)

@torch.no_grad()
def validate(model, test_loader, noise_lvl):

    dataset_size = len(test_loader.dataset)

    total_loss = 0.
    model.eval()

    pbar = tqdm.tqdm(total=len(test_loader), position=0, leave=True, file=sys.stdout)

    for images in test_loader:
        images = images.to(model.device)
        noise = torch.randn_like(images) * noise_lvl / 255.
        noisy = images + noise
        output = model(noisy)
        total_loss += F.mse_loss(output, images).item() * images.size(0)
        pbar.update(1)

    return total_loss / float(dataset_size)


def train(model, optimizer, max_epoch, train_loader, noise_lvl,
          validation=None, scheduler=None, checkpoint_dir=None, max_tolerance=-1):

    best_loss = 99999.
    tolerated = 0

    log = np.zeros([max_epoch, 2], dtype=np.float)

    for e in range(max_epoch):

        print('Epoch #{:d}'.format(e+1))

        log[e, 0] = train_single_epoch(model, optimizer, train_loader, noise_lvl)

        print('\nTrain Loss: {:.5f}'.format(log[e, 0]))

        if scheduler is not None:
            scheduler.step()

        if validation is not None:

            log[e, 1] = validate(model, validation, noise_lvl)

            print('\nVal Loss: {:.5f}'.format(log[e, 1]))

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
