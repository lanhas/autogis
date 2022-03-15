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
from utils import mul_transforms as et
from datasets.mtsd import villageFactorsSegm
from itertools import combinations

# __all__ = ["PairSelector", "AllPositivePairSelector", "HardNegativePairSelector", "TripletSelector", "AllTripletSelector",
#             "FunctionNegativeTripletSelector"]


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


def set_bn_momentum(model, momentum=0.1):
    for m in model.modules():
        if isinstance(m, nn.BatchNorm2d):
            m.momentum = momentum


def fix_bn(model):
    for m in model.modules():
        if isinstance(m, nn.BatchNorm2d):
            m.eval()


def mkdir(path):
    if not os.path.exists(path):
        os.mkdir(path)


def cal_normpara(root_path, image_set):
    """
    计算均值和方差
    """
    transform = et.ExtCompose([
        et.ExtToTensor(),
    ])
    dst = villageFactorsSegm(root_path, image_set=image_set, transform=transform)
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


def pdist(vectors):
    distance_matrix = -2 * vectors.mm(torch.t(vectors)) + vectors.pow(2).sum(dim=1).view(1, -1) + vectors.pow(2).sum(
        dim=1).view(-1, 1)
    return distance_matrix


class PairSelector:
    """
    Implementation should return indices of positive pairs and negative pairs that will be passed to compute
    Contrastive Loss
    return positive_pairs, negative_pairs
    """

    def __init__(self):
        pass

    def get_pairs(self, embeddings, labels):
        raise NotImplementedError


class AllPositivePairSelector(PairSelector):
    """
    Discards embeddings and generates all possible pairs given labels.
    If balance is True, negative pairs are a random sample to match the number of positive samples
    """
    def __init__(self, balance=True):
        super(AllPositivePairSelector, self).__init__()
        self.balance = balance

    def get_pairs(self, embeddings, labels):
        labels = labels.cpu().data.numpy()
        all_pairs = np.array(list(combinations(range(len(labels)), 2)))
        all_pairs = torch.LongTensor(all_pairs)
        positive_pairs = all_pairs[(labels[all_pairs[:, 0]] == labels[all_pairs[:, 1]]).nonzero()]
        negative_pairs = all_pairs[(labels[all_pairs[:, 0]] != labels[all_pairs[:, 1]]).nonzero()]
        if self.balance:
            negative_pairs = negative_pairs[torch.randperm(len(negative_pairs))[:len(positive_pairs)]]

        return positive_pairs, negative_pairs


class HardNegativePairSelector(PairSelector):
    """
    Creates all possible positive pairs. For negative pairs, pairs with smallest distance are taken into consideration,
    matching the number of positive pairs.
    """

    def __init__(self, cpu=True):
        super(HardNegativePairSelector, self).__init__()
        self.cpu = cpu

    def get_pairs(self, embeddings, labels):
        if self.cpu:
            embeddings = embeddings.cpu()
        distance_matrix = pdist(embeddings)

        labels = labels.cpu().data.numpy()
        all_pairs = np.array(list(combinations(range(len(labels)), 2)))
        all_pairs = torch.LongTensor(all_pairs)
        positive_pairs = all_pairs[(labels[all_pairs[:, 0]] == labels[all_pairs[:, 1]]).nonzero()]
        negative_pairs = all_pairs[(labels[all_pairs[:, 0]] != labels[all_pairs[:, 1]]).nonzero()]

        negative_distances = distance_matrix[negative_pairs[:, 0], negative_pairs[:, 1]]
        negative_distances = negative_distances.cpu().data.numpy()
        top_negatives = np.argpartition(negative_distances, len(positive_pairs))[:len(positive_pairs)]
        top_negative_pairs = negative_pairs[torch.LongTensor(top_negatives)]

        return positive_pairs, top_negative_pairs


class TripletSelector:
    """
    Implementation should return indices of anchors, positive and negative samples
    return np array of shape [N_triplets x 3]
    """

    def __init__(self):
        pass

    def get_triplets(self, embeddings, labels):
        raise NotImplementedError


class AllTripletSelector(TripletSelector):
    """
    Returns all possible triplets
    May be impractical in most cases
    """

    def __init__(self):
        super(AllTripletSelector, self).__init__()

    def get_triplets(self, embeddings, labels):
        labels = labels.cpu().data.numpy()
        triplets = []
        for label in set(labels):
            label_mask = (labels == label)
            label_indices = np.where(label_mask)[0]
            if len(label_indices) < 2:
                continue
            negative_indices = np.where(np.logical_not(label_mask))[0]
            anchor_positives = list(combinations(label_indices, 2))  # All anchor-positive pairs

            # Add all negatives for all positive pairs
            temp_triplets = [[anchor_positive[0], anchor_positive[1], neg_ind] for anchor_positive in anchor_positives
                             for neg_ind in negative_indices]
            triplets += temp_triplets

        return torch.LongTensor(np.array(triplets))


def hardest_negative(loss_values):
    hard_negative = np.argmax(loss_values)
    return hard_negative if loss_values[hard_negative] > 0 else None


def random_hard_negative(loss_values):
    hard_negatives = np.where(loss_values > 0)[0]
    return np.random.choice(hard_negatives) if len(hard_negatives) > 0 else None


def semihard_negative(loss_values, margin):
    semihard_negatives = np.where(np.logical_and(loss_values < margin, loss_values > 0))[0]
    return np.random.choice(semihard_negatives) if len(semihard_negatives) > 0 else None


class FunctionNegativeTripletSelector(TripletSelector):
    """
    For each positive pair, takes the hardest negative sample (with the greatest triplet loss value) to create a triplet
    Margin should match the margin used in triplet loss.
    negative_selection_fn should take array of loss_values for a given anchor-positive pair and all negative samples
    and return a negative index for that pair
    """

    def __init__(self, margin, negative_selection_fn, cpu=True):
        super(FunctionNegativeTripletSelector, self).__init__()
        self.cpu = cpu
        self.margin = margin
        self.negative_selection_fn = negative_selection_fn

    def get_triplets(self, embeddings, labels):
        if self.cpu:
            embeddings = embeddings.cpu()
        distance_matrix = pdist(embeddings)
        distance_matrix = distance_matrix.cpu()

        labels = labels.cpu().data.numpy()
        triplets = []

        for label in set(labels):
            label_mask = (labels == label)
            label_indices = np.where(label_mask)[0]
            if len(label_indices) < 2:
                continue
            negative_indices = np.where(np.logical_not(label_mask))[0]
            anchor_positives = list(combinations(label_indices, 2))  # All anchor-positive pairs
            anchor_positives = np.array(anchor_positives)

            ap_distances = distance_matrix[anchor_positives[:, 0], anchor_positives[:, 1]]
            for anchor_positive, ap_distance in zip(anchor_positives, ap_distances):
                loss_values = ap_distance - distance_matrix[torch.LongTensor(np.array([anchor_positive[0]])), torch.LongTensor(negative_indices)] + self.margin
                loss_values = loss_values.data.cpu().numpy()
                hard_negative = self.negative_selection_fn(loss_values)
                if hard_negative is not None:
                    hard_negative = negative_indices[hard_negative]
                    triplets.append([anchor_positive[0], anchor_positive[1], hard_negative])

        if len(triplets) == 0:
            triplets.append([anchor_positive[0], anchor_positive[1], negative_indices[0]])

        triplets = np.array(triplets)

        return torch.LongTensor(triplets)


def HardestNegativeTripletSelector(margin, cpu=False): return FunctionNegativeTripletSelector(margin=margin,
                                                                                 negative_selection_fn=hardest_negative,
                                                                                 cpu=cpu)


def RandomNegativeTripletSelector(margin, cpu=False): return FunctionNegativeTripletSelector(margin=margin,
                                                                                negative_selection_fn=random_hard_negative,
                                                                                cpu=cpu)


def SemihardNegativeTripletSelector(margin, cpu=False): return FunctionNegativeTripletSelector(margin=margin,
                                                                                  negative_selection_fn=lambda x: semihard_negative(x, margin),
                                                                                  cpu=cpu)


if __name__ == "__main__":
    update_testSet('mtvcd')