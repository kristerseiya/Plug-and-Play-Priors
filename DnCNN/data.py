
from torch.utils.data import Dataset
from torchvision import transforms
import glob
from PIL import Image
import os
import numpy as np


def get_transform(mode):
    if mode == 'train':
        transform = transforms.Compose([transforms.RandomCrop((50, 50)),
                                        transforms.RandomVerticalFlip(),
                                        transforms.RandomHorizontalFlip(),
                                        transforms.ColorJitter(),
                                        transforms.ToTensor(),
                                       ])

    elif mode in ['val', 'test']:
        transform = transforms.Compose([transforms.ToTensor()])

    elif mode == 'none':
        transform = lambda x: x

    return transform

class ImageDataSubset(Dataset):
    def __init__(self, dataset, indices, mode):
        self.dataset = dataset
        self.indices = indices
        self.transform = get_transform(mode)

    def set_mode(self, mode):
        self.transform = get_transform(mode)

    def __getitem__(self, idx):
        return self.transform(self.dataset[self.indices[idx]])

    def __len__(self):
        return len(self.indices)


class ImageDataset(Dataset):
    def __init__(self, root_dirs, mode='none', store='ram'):
        super().__init__()
        self.images = list()
        self.store = store

        if type(root_dirs) != list:
            root_dirs = list(root_dirs)

        for root_dir in root_dirs:
            for file_path in glob.glob(os.path.join(root_dir, '*.png')):
                if store == 'ram':
                    fptr = Image.open(file_path).convert('L')
                    file_copy = fptr.copy()
                    fptr.close()
                    self.images.append(file_copy)
                elif store == 'disk':
                    self.images.append(file_path)

        self.transform = get_transform(mode)

    def set_mode(self, mode):
        self.transform = get_transform(mode)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        if self.store == 'ram':
            return self.transform(self.images[idx])
        return self.transform(Image.open(self.images[idx]).convert('L'))

    def split(self, ratios):
        ratios = ratios / np.sum(ratios)
        total_num = len(self.images)
        indices = list(range(total_num))
        np.random.shuffle(indices)

        split1 = int(np.floor(total_num * ratios[0]))
        split2 = int(np.floor(total_num * ratios[1]))

        train_set = ImageDataSubset(self, indices[:split1], 'train')
        val_set = ImageDataSubset(self, indices[split1:split1+split2], 'val')
        test_set = ImageDataSubset(self, indices[split1+split2:], 'test')

        return train_set, val_set, test_set
