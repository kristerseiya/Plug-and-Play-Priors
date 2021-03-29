
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
    def __init__(self, root_dirs, mode='none', store='ram', repeat=1):
        super().__init__()
        self.images = list()
        self.store = store
        self.repeat = repeat

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
        return int(len(self.images) * self.repeat)

    def __getitem__(self, idx):
        if self.store == 'ram':
            return self.transform(self.images[idx // self.repeat])
        return self.transform(Image.open(self.images[idx // self.repeat]).convert('L'))

    def split(self, train_r, val_r, test_r):
        ratios = np.array([train_r, val_r, test_r]) / (train_r + val_r + test_r)
        total_num = len(self.images)
        indices = np.arange(total_num)
        np.random.shuffle(indices)

        split1 = int(np.floor(total_num * ratios[0]))
        split2 = int(np.floor(total_num * ratios[1]))

        train_idx_ = indices[:split1] * self.repeat
        val_idx_ = indices[split1:split1+split2] * self.repeat
        test_idx_ = indices[split1+split2:] * self.repeat

        train_idx = train_idx_
        val_idx = val_idx_
        test_idx = test_idx_

        for i in range(1, self.repeat):
            train_idx = np.concatenate([train_idx, train_idx_ + i])
            val_idx = np.concatenate([val_idx, val_idx_ + i])
            test_idx = np.concatenate([test_idx, test_idx_ + i])

        train_set = ImageDataSubset(self, train_idx, 'train')
        val_set = ImageDataSubset(self, val_idx, 'val')
        test_set = ImageDataSubset(self, test_idx, 'test')

        return train_set, val_set, test_set
