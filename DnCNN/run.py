
import torch
import torch.nn.functional as F
import numpy as np
import torch
import os

def train_single_epoch(model, optimizer, train_loader, noise_lvl):

    dataset_size = len(train_loader.dataset)

    total_loss = 0.
    model.train()
    for images in train_loader:
        optimizer.zero_grad()
        images = images.to(model.device)
        noise = torch.randn_like(images) * noise_lvl / 255.
        noisy = images + noise
        output = model(noisy)
        loss = F.mse_loss(output, noise, reduction='sum')
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    return total_loss / float(dataset_size)

def test(model, test_loader, noise_lvl):

    dataset_size = len(test_loader.dataset)

    with torch.no_grad():
        total_loss = 0.
        model.eval()
        for images, labels in test_loader:
            images = images.to(model.device)
            noise = torch.randn_like(images) * noise_lvl / 255.
            noisy = images + noise
            output = model(noisy)
            total_loss += F.mse_loss(output, noise, reduction='sum').item()

    return total_loss / float(dataset_size)


def train(model, optimizer, max_epoch, train_loader, noise_lvl,
          validation=None, scheduler=None, checkpoint_dir=None, max_tolerance=-1):

    best_loss = 99999.
    tolerated = 0

    log = np.zeros([max_epoch, 2], dtype=np.float)

    for e in range(max_epoch):

        print('Epoch #{:d}'.format(e+1))

        log[e, 0] = train_single_epoch(model, optimizer, train_loader, noise_lvl)

        print('Train Loss: {:.3f}'.format(log[e, 0]))

        if scheduler is not None:
            scheduler.step()

        if validation is not None:

            log[e, 1] = test(model, validation, noise_lvl)

            print('Val Loss: {:.3f}'.format(log[e, 1]))

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

if __name__ == '__main__':

    import argparse
    from torch.utils.data import DataLoader
    from torch.optim import Adam

    from . import data
    from . import model

    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', type=str, default=None)
    parser.add_argument('--data_dir', type=str, required=True)
    parser.add_argument('--n_epoch', type=int, required=True)
    parser.add_argument('--save', type=str, required=True)
    args = parser.parse_args()

    dataset = data.ImageDataset(args.data_dir, 'none')
    trainset, valset, testset = dataset.split(0.7, 0.1, 0.2)
    trainloader = DataLoader(trainset, batch_size=64, shuffle=True, drop_last=True)
    valloader = DataLoader(valset, batch_size=64, shuffle=False)
    testloader = DataLoader(testset, batch_size=64, shuffle=False)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    net = model.DnCNN().move(device)
    if args.weights != None:
        net.load_state_dict(torch.load(args.weights, map_location=device))
    optimizer = Adam(net.parameters(), lr=1e-4)
    train(net, optimizer, args.n_epoch, trainloader, 40, valloader, args.save, -1)

    torch.save(net.state_dict(), os.path.join(args.save, 'final.pth'))
