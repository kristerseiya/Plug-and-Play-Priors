
from torch.utils.data import Dataset
from torchvision import transforms
import glob
from PIL import Image
import os
import numpy as np


def get_transform(mode):
    if mode == 'train':
        transform = transforms.Compose([transforms.RandomVerticalFlip(),
                                        transforms.RandomHorizontalFlip(),
                                        # transforms.ColorJitter(),
                                        transforms.ToTensor(),
                                       ])

    elif mode in 'val':
        transform = transforms.Compose([transforms.ToTensor()])

    elif mode == 'none':
        transform = transforms.Compose([])

    return transform

class Rescale():
    def __init__(self, scale):
        self.scale = scale

    def __call__(self, x):
        new_size = (int(x.size[0] * self.scale), int(x.size[1] * self.scale))
        return transforms.Resize(new_size)(x)

class ImageDataSubset(Dataset):
    def __init__(self, dataset, indices, mode='none', patch_size=-1, repeat=1):
        super().__init__()
        self.dataset = dataset
        self.indices = indices
        self.repeat = repeat
        self.patch_size = patch_size
        self.transform = get_transform(mode)
        self.patch_size = patch_size
        if (self.patch_size > 0):
            self.transform.transforms.insert(0, transforms.RandomCrop(self.patch_size))
        if (self.dataset.scale > 0) and (self.dataset.store == 'disk'):
            self.transform.transforms.insert(0, Rescale(self.dataset.scale))

    def set_mode(self, mode):
        self.transform = get_transform(mode)
        if (self.patch_size > 0):
            self.transform.transforms.insert(0, transforms.RandomCrop(self.patch_size))
        if (self.dataset.scale > 0) and (self.dataset.store == 'disk'):
            self.transform.transforms.insert(0, Rescale(self.dataset.scale))

    def set_patch(self, patch_size):
        if (self.patch_size > 0):
            if (self.dataset.scale > 0) and (self.dataset.store == 'disk'):
                self.transform.transforms.pop(1)
            else:
                self.transform.transforms.pop(0)
        self.patch_size = patch_size
        if (patch_size > 0):
            if (self.dataset.scale > 0) and (self.dataset.store == 'disk'):
                self.transform.transforms.insert(1, transforms.RandomCrop(patch_size))
            else:
                self.transform.transforms.insert(0, transforms.RandomCrop(patch_size))

    def __getitem__(self, idx):
        if self.dataset.store == 'ram':
            return self.transform(self.dataset.images[self.indices[idx // self.repeat]])
        return self.transform(Image.open(self.dataset.images[self.indices[idx // self.repeat]]).convert('L'))

    def __len__(self):
        return len(self.indices) * self.repeat

class ImageDataset(Dataset):
    def __init__(self, root_dirs, mode='none', scale=-1, patch_size=-1, repeat=1, store='ram'):
        super().__init__()
        self.images = list()
        self.store = store
        self.repeat = repeat
        self.scale = scale
        self.patch_size = patch_size

        if type(root_dirs) != list:
            root_dirs = [root_dirs]

        for root_dir in root_dirs:
            for file_path in glob.glob(os.path.join(root_dir, '*.png')):
                if store == 'ram':
                    fptr = Image.open(file_path).convert('L')
                    file_copy = fptr.copy()
                    fptr.close()
                    if scale > 0:
                        file_copy = Rescale(scale)(file_copy)
                    self.images.append(file_copy)
                elif store == 'disk':
                    self.images.append(file_path)

        self.transform = get_transform(mode)
        if (self.patch_size > 0):
            self.transform.transforms.insert(0, transforms.RandomCrop(self.patch_size))
        if (self.scale > 0) and (self.store == 'disk'):
            self.transform.transforms.insert(0, Rescale(self.scale))

    def set_mode(self, mode):
        self.transform = get_transform(mode)
        if (self.patch_size > 0):
            self.transform.transforms.insert(0, transforms.RandomCrop(self.patch_size))
        if (self.scale > 0) and (self.store == 'disk'):
            self.transform.transforms.insert(0, Rescale(self.scale))

    def set_patch(self, patch_size):
        if (self.patch_size > 0):
            if (self.scale > 0) and (self.store == 'disk'):
                self.transform.transforms.pop(1)
            else:
                self.transform.transforms.pop(0)
        self.patch_size = patch_size
        if (patch_size > 0):
            if (self.scale > 0) and (self.store == 'disk'):
                self.transform.transforms.insert(1, transforms.RandomCrop(patch_size))
            else:
                self.transform.transforms.insert(0, transforms.RandomCrop(patch_size))

    def __len__(self):
        return int(len(self.images) * self.repeat)

    def __getitem__(self, idx):
        if self.store == 'ram':
            return self.transform(self.images[idx // self.repeat])
        return self.transform(Image.open(self.images[idx // self.repeat]).convert('L'))

    def split(self, *r):
        ratios = np.array(r)
        ratios = ratios / ratios.sum()
        total_num = len(self.images)
        indices = np.arange(total_num)
        np.random.shuffle(indices)

        subsets = list()
        start = 0
        for r in ratios[:-1]:
            split = int(total_num * r)
            subsets.append(ImageDataSubset(self, indices[start:start+split]))
            start = start + split
        subsets.append(ImageDataSubset(self, indices[start:]))

        return subsets
