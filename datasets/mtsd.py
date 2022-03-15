import numpy as np
import torch.utils.data as data
from pathlib import Path
from PIL import Image


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


class villageFactorsSegm(data.Dataset):
    cmap = village_cmap()
    def __init__(self, root_mtsd, image_set='train', transform=None) -> None:
        self.root_mtsd = Path(root_mtsd)
        self.transform = transform
        self.image_set = image_set
        image_dir = self.root_mtsd / 'JPEGImages'
        dem_dir = self.root_mtsd / 'DEMImages'
        mask_dir = self.root_mtsd / 'SegmentationClass'
        split_f = self.root_mtsd / 'ImageSets/Segmentation' / (image_set + '.txt')
        if not split_f.exists():
            raise ValueError(
                'Wrong image_set entered! Please use image_set="train" '
                'or image_set="trainval" or image_set="val"')

        with open(split_f, "r") as f:
            file_names = [x.strip() for x in f.readlines()]

        self.images = [image_dir / (x + '.jpg') for x in file_names]
        self.dems = [dem_dir / (x + '.jpg') for x in file_names]
        self.masks = [mask_dir / (x + ".png") for x in file_names]
        assert (len(self.images) == len(self.masks) and len(self.dems) == len(self.masks))

    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: (image, target) where target is the image segmentation.
        """
        img = Image.open(self.images[index]).convert('RGB')
        dem = Image.open(self.dems[index]).convert('L')
        target = Image.open(self.masks[index])
        if self.transform is not None:
            img, dem, target = self.transform(img, dem, target)
        return img, dem, target

    def __len__(self):
        return len(self.images)

    @classmethod
    def decode_target(cls, mask):
        """decode semantic mask to RGB image"""
        return cls.cmap[mask]


if __name__ == "__main__":
    print(__package__)