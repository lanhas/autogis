import os
import numpy as np
from PIL import Image
from pathlib import Path
from torchvision import transforms
from torch.utils.data.dataset import Dataset
from torch.utils.data.sampler import BatchSampler

from .datasets import register


@register('village-clss')
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

        image_size = 256
        data = [Image.open(x) for x in img_names]

        min_label = min(img_label)
        label = [x - min_label for x in img_label]

        self.data = data
        self.label = label
        self.n_classes = max(self.label) + 1

        self.default_transform = transforms.Compose([
            transforms.Resize(image_size),
            transforms.RandomRotation(180),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
        ])

        self.transform = self.default_transform

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, index):
        return self.transform(self.data[index]), self.label[index]


class VillageSiamese(Dataset):
    """
    Train:
    Test:
    """
    def __init__(self, root_path, split="train", transform=None) -> None:
        self.root_path = Path(root_path)
        self.transform = transform
        self.split = split
        image_dir = self.root_path / "JPEGImages"
        split_f = self.root_path / "ImageSets" / (split + '.txt')
        if not os.path.exists(split_f):
            raise ValueError(
                'Wrong split entered! Please use split="train" '
                'or split="trainval" or split="val"')

        data = np.loadtxt(split_f, dtype=str, delimiter=',')
        file_names = list(data[:, 0])

        label = np.array(data[:, 1])
        min_label = min(label)
        label = [x - min_label for x in label]

        self.data = [os.path.join(image_dir, x + ".png") for x in file_names]
        self.label = label
        self.labels_set = set(np.array(data[:, 1]))
        self.label_to_indices = {
            label: np.where(self.labels == label)[0] 
            for label in self.labels_set}

        if self.split == "val":
            random_state = np.random.RandomState(29)

            positive_pairs = [[i,
                              random_state.choice(self.label_to_indices[self.labels[i].item()]),
                              1]
                              for i in range(0, len(self.data), 2)]
            negative_pairs = [[i,
                              random_state.choice(self.label_to_indices[
                                                    np.random.choice(
                                                        list(self.labels_set - set([self.labels[i].item()]))
                                                    )
                                                ]),
                              0]
                              for i in range(1, len(self.data), 2)]
            self.test_pairs = positive_pairs + negative_pairs
    
    def __getitem__(self, index):
        if self.split == "train":
            target = np.random.randint(0, 2)
            img1, label1 = self.data[index], self.labels[index].item()
            if target == 1:
                siamese_index = index
                while siamese_index == index:
                    siamese_index = np.random.choice(self.label_to_indices[label1])
            else:
                siamese_label = np.random.choice(list(self.labels_set - set([label1])))
                siamese_index = np.random.choice(self.label_to_indices[siamese_label])
            img2 = self.data[siamese_index]
        else:
            img1 = self.data[self.test_pairs[index][0]]
            img2 = self.data[self.test_pairs[index][1]]
            target = self.test_pairs[index][2]

        img1 = Image.open(img1)
        img2 = Image.open(img2)
        if self.transform is not None:
            img1 = self.transform(img1)
            img2 = self.transform(img2)
        return (img1, img2), target

    def get_labels(self):
        return self.labels

    def get_datas(self):
        return self.data

    def __len__(self):
        return len(self.data)


class VillageTriplet(Dataset):
    """
    Train: For each sample (anchor) randomly chooses a positive and negative samples
    Test: Creates fixed triplets for testing
    """
    def __init__(self, root_path, split="train", transform=None) -> None:
        self.root_path = Path(root_path)
        self.transform = transform
        self.split = split
        image_dir = self.root_path / "JPEGImages"
        split_f = self.root_path / "ImageSets" / (split + '.txt')
        if not os.path.exists(split_f):
            raise ValueError(
                'Wrong split entered! Please use split="train" '
                'or split="trainval" or split="val"')

        data = np.loadtxt(split_f, dtype=str, delimiter=',')
        file_names = list(data[:, 0])

        self.labels = np.array(data[:, 1])
        self.data = [os.path.join(image_dir, x + ".png") for x in file_names]
        self.labels_set = set(np.array(data[:, 1]))
        self.label_to_indices = {
            label: np.where(self.labels == label)[0] 
            for label in self.labels_set}

        if self.split == "val":
            random_state = np.random.RandomState(29)

            triplets = [[i,
                         random_state.choice(self.label_to_indices[self.labels[i].item()]),
                         random_state.choice(self.label_to_indices[
                                                np.random.choice(
                                                    list(self.labels_set - set([self.labels[i].item()]))
                                                )
                         ])
                        ]
                        for i in range(len(self.data))]
            self.test_triplets = triplets
    
    def __getitem__(self, index):
        if self.split == "train":
            img1, label1 = self.data[index], self.labels[index].item()
            positive_index = index
            while positive_index == index:
                positive_index = np.random.choice(self.label_to_indices[label1])
            negative_label = np.random.choice(list(self.labels_set - set([label1])))
            negative_index = np.random.choice(self.label_to_indices[negative_label])
            img2 = self.data[positive_index]
            img3 = self.data[negative_index]
        else:
            img1 = self.data[self.test_triplets[index][0]]
            img2 = self.data[self.test_triplets[index][1]]
            img3 = self.data[self.test_triplets[index][2]]
        img1 = Image.open(img1)
        img2 = Image.open(img2)
        img3 = Image.open(img3)
        if self.transform is not None:
            img1 = self.transform(img1)
            img2 = self.transform(img2)
            img3 = self.transform(img3)
        return (img1, img2, img3), []

    def get_labels(self):
        return self.labels

    def get_datas(self):
        return self.data

    def __len__(self):
        return len(self.data)


class BalancedBatchSampler(BatchSampler):
    def __init__(self, labels, n_classes, n_samples) -> None:
        self.labels = labels
        self.labels_set = list(set(self.labels))
        self.label_to_indices = {label: np.where(self.labels == label)[0] 
                                 for label in self.labels_set}
        for l in self.labels_set:
            np.random.shuffle(self.label_to_indices[l])
        self.used_label_indices_count = {label: 0 for label in self.labels_set}
        self.count = 0
        self.n_classes = n_classes
        self.n_samples = n_samples
        self.n_dataset = len(self.labels)
        self.batch_size = self.n_samples * self.n_classes

    def __iter__(self):
        self.count = 0
        while self.count + self.batch_size < self.n_dataset:
            classes = np.random.choice(self.labels_set, self.n_classes, replace=False)
            indices = []
            for class_ in classes:
                indices.extend(self.label_to_indices[class_][self.used_label_indices_count[class_]:self.used_label_indices_count[class_] + self.n_samples])
                self.used_label_indices_count[class_] += self.n_samples
                if self.used_label_indices_count[class_] + self.n_samples > len(self.label_to_indices[class_]):
                    np.random.shuffle(self.label_to_indices[class_])
                    self.used_label_indices_count[class_] = 0
            yield indices
            self.count += self.n_classes * self.n_samples
            
    def __len__(self):
        return self.n_dataset // self.batch_size

