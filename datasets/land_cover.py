import torch
import torch.utils.data as data
from PIL import Image
from pathlib import Path

from .datasets import register
from utils import ext_transforms as et
from utils.utils import color2annotation


@register('land-cover')
class RoadSegm(data.Dataset):
    """deep globe dataset"""
    def __init__(self, root_path, split='train', **kwargs):
        root_path = Path(root_path)
        self.sat_img_names = list(filter(lambda x: '_sat' in str(x), (root_path / split).iterdir()))
        self.mask_img_names = [Path(str(sat_name).split('_')[0] + '_mask.png') for sat_name in self.sat_img_names]
        # self.mask_img_names = list(filter(lambda x: '_mask' in str(x), (root_path / split).iterdir()))
        # assert (len(self.sat_img_names) == len(self.mask_img_names))

        norm_params = {'mean': [],
                       'std': []}
        self.n_classes = 7
        crop_size = 512
        self.train_transform = et.ExtCompose([
            et.ExtRandomScale((0.5, 2.0)),
            et.ExtRandomCrop(size=(crop_size, crop_size), pad_if_needed=True),
            et.ExtRandomHorizontalFlip(),
            et.ExtToTensor(),
        ])

        self.val_transform = et.ExtCompose([
            et.ExtToTensor(),
        ])

        if split == "train":
            self.transform = self.train_transform
        else:
            self.transform = self.val_transform

        def convert_raw(x):
            mean = torch.tensor(norm_params['mean']).view(3, 1, 1).type_as(x)
            std = torch.tensor(norm_params['std']).view(3, 1, 1).type_as(x)
            return x * std + mean
        self.convert_raw = convert_raw

    def __getitem__(self, index):
        image = Image.open(self.sat_img_names[index])
        label = Image.open(self.mask_img_names[index])
        label = color2annotation(label)
        return self.transform(image, label)

    def __len__(self):
        return len(self.sat_img_names)
