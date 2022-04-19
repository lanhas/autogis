
import os
import numpy as np
from PIL import Image
from pathlib import Path
import torch.utils.data as data


class RoadSegm(data.Dataset):
    """deep globe dataset"""
    def __init__(self, root_dir, status='train', transform=None):
        self.status = status
        self.root_dir = Path(root_dir)
        self.transform = transform
        self.sat_img_names = list(filter(lambda x: '_sat' in x, os.listdir(self.root_dir / self.status)))

    def __getitem__(self, index):
        sat_img_nm = self.sat_img_names[index]
        mask_img_nm = self.sat_img_names[index].split('_')[0] + '_mask.png'

        sat_img_path = self.root_dir / self.status / sat_img_nm
        mask_img_path = self.root_dir / self.status / mask_img_nm

        sat_img = Image.open(sat_img_path).convert('RGB')
        mask_img = Image.open(mask_img_path).convert('L')

        mask = np.zeros_like(mask_img, np.uint8)
        # since it is not exactly 255 for road area, binarize at 128
        # mask[np.where(np.all(mask_img==(255,255,255), axis=-1))] = 1
        mask[np.array(mask_img, np.uint8) >= 128] = 1
        mask = Image.fromarray(mask).convert('L')
        # sample = {'sat_img': sat_img, 'map_img': mask}
        
        if self.transform:
            sample = self.transform(sat_img, mask)

        return sample

    def __len__(self):
        return len(self.sat_img_names)
