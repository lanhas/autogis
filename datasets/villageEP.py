import os
import torch
import numpy as np
from PIL import Image
from pathlib import Path
from torchvision import transforms
from torch.utils.data.dataset import Dataset
# from utils import ext_transforms as et
from utils.village_segm import mul_transforms as mt
from .datasets import register


def village_cmap(N=7, normalized=False):
    dtype = 'float32' if normalized else 'uint8'
    cmap = np.zeros((N, 3), dtype=dtype)
    cmap[0] = [0, 0, 0]         # 未知土地：黑色，云层遮盖或其他因素
    cmap[1] = [128, 0, 0]       # 山体：地形上突出的部分，包括荒山、林山、农山等
    cmap[2] = [0, 128, 0]       # 森林/植被： 绿色，平原地区且被绿色包裹的非农田区域
    cmap[3] = [128, 128, 0]     # 农田：黄色，平原地区的农业区域
    cmap[4] = [0, 0, 128]       # 水系：深蓝色，江河湖海湿地
    cmap[5] = [128, 0, 128]     # 荒地：紫色
    cmap[6] = [0, 128, 128]     # 村落：浅蓝色，村庄
    cmap = cmap / 255 if normalized else cmap
    return cmap


@register('villageEP-segm')
class VillageSegm(Dataset):
    cmap = village_cmap()

    def __init__(self, root_path, split='train', **kwargs) -> None:
        self.root_path = Path(root_path)
        self.split = split
        image_dir = self.root_path / 'JPEGImages'
        dem_dir = self.root_path / 'DEMImages'
        # image_dir = self.root_path / 'RemoteData'
        # dem_dir = self.root_path / 'ElevationData'
        mask_dir = self.root_path / 'SegmentationClass'
        split_f = self.root_path / 'ImageSets/Segmentation' / (split + '.txt')
        if not split_f.exists():
            raise ValueError(
                'Wrong split entered! Please use split="train" '
                'or split="trainval" or split="val"')

        with open(split_f, "r") as f:
            village_names = [x.strip() for x in f.readlines()]

        images = [image_dir / (x + '.jpg') for x in village_names]
        dems = [dem_dir / (x + '.jpg') for x in village_names]
        # images = [image_dir / (x + '.tif') for x in village_names]
        # dems = [dem_dir / (x + '.tif') for x in village_names]
        masks = [mask_dir / (x + ".png") for x in village_names]

        self.data_r = [Image.open(x) for x in images]
        self.data_d = [Image.open(x) for x in dems]
        self.label = [Image.open(x) for x in masks]

        assert (len(self.data_r) == len(self.data_d) and len(self.data_d) == len(self.label))

        self.n_classes = 7
        crop_size = 512
        norm_params = {'mean_r': [],
                       'std_r': [],
                       'mean_d': [],
                       'std_d': []}
        normalize = mt.ExtNormalize(**norm_params)
        self.default_transform = mt.ExtCompose([
            mt.ExtRandomScale((0.5, 2.0)),
            mt.ExtRandomCrop(size=(crop_size, crop_size), pad_if_needed=True),
            mt.ExtRandomHorizontalFlip(),
            mt.ExtToTensor(),
            # normalize,
        ])
        self.val_transform = mt.ExtCompose([
            # mt.ExtRandomCrop(size=(crop_size, crop_size), pad_if_needed=True),
            mt.ExtRandomCrop(size=(crop_size, crop_size), pad_if_needed=True),
            mt.ExtToTensor(),
        ])
        augment = kwargs.get('augment')
        if self.split == "train":
            self.transform = self.default_transform
        else:
            self.transform = self.val_transform

        def convert_raw(x):
            mean = torch.tensor(norm_params['mean']).view(3, 1, 1).type_as(x)
            std = torch.tensor(norm_params['std']).view(3, 1, 1).type_as(x)
            return x * std + mean
        self.convert_raw = convert_raw

    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: (image, target) where target is the image segmentation.
        """
        return self.transform(self.data_r[index].convert('RGB'), self.data_d[index].convert('L'), self.label[index])

    def __len__(self):
        return len(self.label)

    @classmethod
    def decode_target(cls, mask):
        """decode semantic mask to RGB image"""
        return cls.cmap[mask]


@register('villageEP-clss')
class VillageClss(Dataset):
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

        image_size = 128
        data = [Image.open(x) for x in img_names]

        min_label = min(img_label)
        label = [x - min_label for x in img_label]

        self.data = data
        self.label = label
        self.n_classes = max(self.label) + 1

        self.default_transform = transforms.Compose([
            transforms.Resize(image_size),
            # transforms.RandomRotation(180),
            transforms.ToTensor(),
        ])

        self.transform = self.default_transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        return self.transform(self.data[index]), self.label[index]
