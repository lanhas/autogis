import os
import torch
import numpy as np
import torch.nn as nn
from PIL import Image
from pathlib import Path
from shutil import copyfile
from torch.utils.data import DataLoader
from torchvision.transforms.functional import normalize
from sklearn.model_selection import train_test_split
from utils.village_segm import mul_transforms as et
from datasets.village_segm import VillageSegm
from itertools import combinations


def color2annotation(image):
    image = np.array(image)
    image = (image >= 128).astype(np.uint8)
    image = 4 * image[:, :, 0] + 2 * image[:, :, 1] + image[:, :, 2]
    cat_image = np.zeros((2448, 2448), dtype=np.uint8)
    cat_image[image == 3] = 0  # (Cyan: 011) Urban land
    cat_image[image == 6] = 1  # (Yellow: 110) Agriculture land
    cat_image[image == 5] = 2  # (Purple: 101) Rangeland
    cat_image[image == 2] = 3  # (Green: 010) Forest land
    cat_image[image == 1] = 4  # (Blue: 001) Water
    cat_image[image == 7] = 5  # (White: 111) Barren land
    cat_image[image == 0] = 6  # (Black: 000) Unknown
    res = Image.fromarray(cat_image)
    return res


def annotation2color(image):
    image = np.array(image)
    color = np.zeros((*(2448, 2448), 3), dtype=np.uint8)
    color[image == 0] = [0, 255, 255]         # 人造建筑
    color[image == 1] = [255, 255, 0]         # 农田
    color[image == 2] = [255, 0, 255]         # 牧场
    color[image == 3] = [0, 255, 0]           # 森林
    color[image == 4] = [0, 0, 255]           # 水系
    color[image == 5] = [255, 255, 255]       # 荒地
    color[image == 6] = [0, 0, 0]             # 未知区域
    res = Image.fromarray(color)
    return res


def denormalize(tensor, mean, std):
    """
    反归一化
    """
    mean = np.array(mean)
    std = np.array(std)
    _mean = -mean/std
    _std = 1/std
    return normalize(tensor, _mean, _std)


class Denormalize(object):
    def __init__(self, mean, std):
        mean = np.array(mean)
        std = np.array(std)
        self._mean = -mean/std
        self._std = 1/std

    def __call__(self, tensor):
        if isinstance(tensor, np.ndarray):
            return (tensor - self._mean.reshape(-1,1,1)) / self._std.reshape(-1,1,1)
        return normalize(tensor, self._mean, self._std)



def cal_normpara(root_path, image_set):
    """
    计算均值和方差
    """
    transform = et.ExtCompose([
        et.ExtToTensor(),
    ])
    dst = VillageSegm(root_path, image_set=image_set, transform=transform)
    loader = DataLoader(dataset=dst, batch_size=32, shuffle=True)
    channels_sum_image, channels_sum_dem, channel_squared_sum_image, channel_squared_sum_dem, num_batches = 0, 0, 0, 0, 0
    for data_image, data_dem, target in loader:
        channels_sum_image += torch.mean(data_image, dim=[0, 2, 3])
        channel_squared_sum_image += torch.mean(data_image ** 2, dim=[0, 2, 3])
        channels_sum_dem += torch.mean(data_dem, dim=[0, 2, 3])
        channel_squared_sum_dem += torch.mean(data_dem ** 2, dim=[0, 2, 3])
        num_batches += 1
    mean_image = channels_sum_image / num_batches
    mean_dem = channels_sum_dem / num_batches
    std_image = (channel_squared_sum_image / num_batches - mean_image ** 2) ** 0.5
    std_dem = (channel_squared_sum_dem / num_batches - mean_dem ** 2) ** 0.5
    print("{}_main_image:{}".format(image_set, mean_image))
    print("{}_std_image:{}".format(image_set, std_image))

    print("{}_main_dem:{}".format(image_set, mean_dem))
    print("{}_std_dem:{}".format(image_set, std_dem))


def extrate_img(file_path, source_paths, target_paths):
    """
    根据val.txt文件中的内容，将jpeg，dem和label文件提取到testSet中
    """
    # 清理缓存，删除旧文件
    for target_path in target_paths:
        for fileName in target_path.iterdir():
            fileName.unlink()
    # 检查文件是否存在
    if not os.path.exists(file_path):
        raise ValueError("文件不存在 请检查后重试！")
    for source_path, target_path in zip(source_paths, target_paths):
        if not (os.path.exists(source_path) and os.path.exists(target_path)):
            raise ValueError("({},{})文件夹不存在 请创建后重试！".format(source_path, target_path))
    # 读取val.txt中的文件
    with open(file_path, 'r') as f:
        file_names = [x.strip() for x in f.readlines()]
    # 复制文件
    for source_path, target_path in zip(source_paths, target_paths):
        for _, _, files in os.walk(source_path):
            firstfile = sorted(files)[0]
        file_suffix = os.path.splitext(firstfile)[1]
        for fileName in file_names:
            source_file = source_path / (fileName + file_suffix)
            target_file = target_path / (fileName + file_suffix)
            copyfile(source_file, target_file)


def update_testSet(dataset_name='mtvcd'):
    """
    更新测试集文件，包括jpeg，dem和label文件
    """
    file_path = Path.cwd() / 'datasets/tarin/mtsd_voc/ImageSets/Segmentation/val.txt'
    mtvcd_total = Path.cwd() / 'datasets/mtvcd/labels/total.csv'
    source_ImgPath = Path.cwd() / 'datasets/tarin/mtsd_voc/JPEGImages'
    source_DemPath = Path.cwd() / 'datasets/tarin/mtsd_voc/DEMImages'
    source_Mask = Path.cwd() / 'datasets/tarin/mtsd_voc/SegmentationClass'
    target_ImgPath = Path.cwd() / 'datasets/test/mtsd_voc/JPEGImages'
    target_DemPath = Path.cwd() / 'datasets/test/mtsd_voc/DEMIMages'
    target_Mask = Path.cwd() / 'datasets/test/mtsd_voc/SegmentationClass'
    target_mtvcd_trainval = Path.cwd() / 'datasets/mtvcd/labels/trainval.txt'
    target_mtvcd_train = Path.cwd() / 'datasets/mtvcd/labels/train.txt'
    target_mtvcd_test = Path.cwd() / 'datasets/mtvcd/labels/val.txt'

    if dataset_name == 'mtsd':
        extrate_img(file_path, (source_ImgPath, source_DemPath, source_Mask), (target_ImgPath, target_DemPath, target_Mask))
    elif dataset_name == 'mtvcd':
        from sklearn.model_selection import train_test_split
        total = np.loadtxt(mtvcd_total, delimiter=',', dtype=str)
        trainval = total[total[:, 1]!='0']
        train, val = train_test_split(trainval, test_size=0.2, shuffle=True)
        np.savetxt(target_mtvcd_trainval, trainval, delimiter=',',fmt = '%s')
        np.savetxt(target_mtvcd_train, train, delimiter=',',fmt = '%s')
        np.savetxt(target_mtvcd_test, val, delimiter=',',fmt = '%s')


def images_append(source_images, source_results, target_folder):
    """
    图片融合
    """
    for fileName_image, fileName_mask in zip(source_images.iterdir(), source_results.iterdir()):
        image = Image.open(fileName_image).convert('RGB')
        mask = Image.open(fileName_mask).convert('RGB')
        result = Image.blend(image, mask, 0.6)
        result_path = target_folder / fileName_image.name
        result.save(result_path, quality=95)
        print(fileName_image.name)


def compose_data():
    """
    融合图片生成结果
    """
    source_images = Path.cwd() / 'datasets/test/mtsd_voc/JPEGImages'
    source_results = Path.cwd() / 'datasets/test/mtsd_voc/Result'
    target_Images = Path.cwd() / 'datasets/test/mtsd_voc/ComposeImages'
    images_append(source_images, source_results, target_Images)


def imageSets_mtsd():
    segfilepath = Path.cwd() / 'datasets/tarin/mtsd_voc/SegmentationClass'
    saveBasePath = Path.cwd() / 'datasets/tarin/mtsd_voc/ImageSets/Segmentation'

    total_seg = []
    for fileName in segfilepath.iterdir():
        if fileName.suffix == ".png":
            total_seg.append(fileName.stem)

    trainval_percent=1.0      # 训练验证集/测试集比例
    train_percent=0.85      # 训练集/验证集比例

    total = np.array(total_seg, dtype=str)
    if trainval_percent == 1.0:
        trainval = total
        test = []
    else:
        trainval, test = train_test_split(total, train_size=trainval_percent)
    train, valid = train_test_split(trainval, train_size=train_percent)
    np.savetxt(saveBasePath / 'trainval.txt', trainval, fmt='%s', delimiter=',')
    np.savetxt(saveBasePath / 'train.txt', train, fmt='%s', delimiter=',')
    np.savetxt(saveBasePath / 'test.txt', test, fmt='%s', delimiter=',')
    np.savetxt(saveBasePath / 'val.txt', valid, fmt='%s', delimiter=',')
    print("train and val size:", len(trainval))
    print("train size:", len(train))


def imageSets_mtvcd():
    total_path = 'datasets/tarin/mtvcd/total.txt'
    saveBasePath = Path.cwd() / 'datasets/tarin/mtvcd'

    trainval_percent=1.0      # 训练验证集/测试集比例
    train_percent=0.85      # 训练集/验证集比例

    total = np.loadtxt(total_path, dtype=str, delimiter=',')
    if trainval_percent == 1.0:
        trainval = total
        test = []
    else:
        trainval, test = train_test_split(total, train_size=trainval_percent)
    train, valid = train_test_split(trainval, train_size=train_percent)
    np.savetxt(saveBasePath / 'trainval.txt', trainval, fmt='%s', delimiter=',')
    np.savetxt(saveBasePath / 'train.txt', train, fmt='%s', delimiter=',')
    np.savetxt(saveBasePath / 'test.txt', test, fmt='%s', delimiter=',')
    np.savetxt(saveBasePath / 'val.txt', valid, fmt='%s', delimiter=',')
    print("train and val size:", len(trainval))
    print("train size:", len(train))


if __name__ == "__main__":
    update_testSet('mtvcd')