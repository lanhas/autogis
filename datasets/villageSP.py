import os
import numpy as np
from PIL import Image
from pathlib import Path
from torchvision import transforms
from torch.utils.data.dataset import Dataset
from utils import ext_transforms as et

from .datasets import register


@register('villageSP-clss')
class VillageClss(Dataset):
    """
    Train:
    test:
    """

    def __init__(self, root_path, split="train", **kwargs) -> None:
        self.root_path = Path(root_path)
        self.split = split
        image_dir = self.root_path / "remote"
        el_dir = self.root_path / "contourLine_g"
        split_f = self.root_path / "ImageSets/pattern" / (split + '.txt')
        if not os.path.exists(split_f):
            raise ValueError(
                'Wrong split entered! Please use split="train" '
                'or split="trainval" or split="val"')

        img_sets = np.loadtxt(split_f, dtype=str, delimiter=',')

        img_names = [image_dir / (x + '_remote.tif') for x in list(img_sets[:, 0])]
        dem_names = [el_dir / (img_name.name[:-11] + '_demi.png') for img_name in img_names]
        img_label = [int(x) for x in list(img_sets[:, 1])]

        image_size = 128
        self.data_remote = [Image.open(x) for x in img_names]
        self.data_dem = [Image.open(x) for x in dem_names]

        min_label = min(img_label)
        label = [x - min_label for x in img_label]

        self.label = label
        self.n_classes = max(self.label) + 1

        self.default_transform = et.ExtCompose([
            et.ExtResize(image_size),
            et.ExtRandomRotation(180),
            et.ExtToTensor(),
            # normalize,
        ])

        self.transform = self.default_transform

    def __len__(self):
        return len(self.data_remote)

    def __getitem__(self, index):
        return *self.transform(self.data_remote[index], self.data_dem[index]), self.label[index]


@register('villageSP-single')
class VillageClssS(Dataset):
    """
    Train:
    test:
    """

    def __init__(self, root_path, split="train", **kwargs) -> None:
        self.root_path = Path(root_path)
        self.split = split
        image_dir = self.root_path / "JPEGImages"
        split_f = self.root_path / "ImageSets" / (split + '.txt')
        if not os.path.exists(split_f):
            raise ValueError(
                'Wrong split entered! Please use split="train" '
                'or split="trainval" or split="val"')

        img_sets = np.loadtxt(split_f, dtype=str, delimiter=',')
        img_names = [os.path.join(image_dir, x + ".png") for x in list(img_sets[:, 0])]
        img_label = [int(x) for x in list(img_sets[:, 1])]

        image_size = 512
        data = [Image.open(x) for x in img_names]

        min_label = min(img_label)
        label = [x - min_label for x in img_label]

        self.data = data
        self.label = label
        self.n_classes = max(self.label) + 1

        self.default_transform = transforms.Compose([
            transforms.Resize(image_size),
            transforms.RandomRotation(180),
            transforms.ToTensor(),
        ])

        self.transform = self.default_transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        return self.transform(self.data[index]), self.label[index]
