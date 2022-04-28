import torch
import torch.utils.data as data
from PIL import Image
from pathlib import Path
from torchvision import transforms

from .datasets import register
from utils import ext_transforms as et


@register('land-cover')
class RoadSegm(data.Dataset):
    """deep globe dataset"""
    def __init__(self, root_path, split='train', **kwargs):
        root_path = Path(root_path)
        self.sat_img_names = list(filter(lambda x: '_sat' in str(x), (root_path / split).iterdir()))
        self.mask_img_names = list(filter(lambda x: '_mask' in str(x), (root_path / split).iterdir()))
        assert (len(self.sat_img_names) == len(self.mask_img_names))

        norm_params = {'mean': [],
                       'std': []}
        self.n_classes = 7
        crop_size = 512
        normalize = transforms.Normalize(**norm_params)
        self.default_transform = et.ExtCompose([
            et.ExtRandomScale((0.5, 2.0)),
            et.ExtRandomCrop(size=(crop_size, crop_size), pad_if_needed=True),
            et.ExtRandomHorizontalFlip(),
            et.ExtToTensor(),
        ])
        augment = kwargs.get('augment')
        if augment is None:
            self.transform = self.default_transform

        def convert_raw(x):
            mean = torch.tensor(norm_params['mean']).view(3, 1, 1).type_as(x)
            std = torch.tensor(norm_params['std']).view(3, 1, 1).type_as(x)
            return x * std + mean
        self.convert_raw = convert_raw

    def __getitem__(self, index):
        image = Image.open(self.sat_img_names[index])
        label = Image.open(self.mask_img_names[index])
        return self.transform(image, label.convert('1'))

    def __len__(self):
        return len(self.sat_img_names)
