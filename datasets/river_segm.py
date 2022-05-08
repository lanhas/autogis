import torch
import numpy as np
from torch.utils.data import Dataset
from pathlib import Path
from PIL import Image
from utils.village_segm import mul_transforms as mt

from .datasets import register


@register('river-segm')
class VillageSegm(Dataset):

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

        self.n_classes = 2
        crop_size = 256
        # val_crop_size = 1536
        val_crop_size = 256
        norm_params = {'mean_r': [],
                       'std_r': [],
                       'mean_d': [],
                       'std_d': []}
        normalize = mt.ExtNormalize(**norm_params)
        train_transform = mt.ExtCompose([
            mt.ExtRandomScale((0.5, 2.0)),
            mt.ExtRandomCrop(size=(crop_size, crop_size), pad_if_needed=True),
            mt.ExtRandomHorizontalFlip(),
            mt.ExtToTensor(),
        ])

        val_transform = mt.ExtCompose([
            mt.ExtCenterCrop(size=(val_crop_size, val_crop_size)),
            mt.ExtToTensor(),
        ])

        if self.split == "train":
            self.transform = train_transform
        else:
            self.transform = val_transform

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
        return self.transform(self.data_r[index], self.data_d[index], self.label[index])

    def __len__(self):
        return len(self.label)

    @classmethod
    def decode_target(cls, mask):
        """decode semantic mask to RGB image"""
        return cls.cmap[mask]
