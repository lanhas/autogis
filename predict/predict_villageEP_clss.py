import os
import torch
import numpy as np
import models.village_clss as models

from torch.utils import data
from datasets.villageEP import VillageClss
import torchvision.transforms as T
import matplotlib.pyplot as plt

village_classes = ['mountain ring of water around', 'adjoin mountain', 'along river', 'plain',
                   'mountain']  # ['山环水绕型', '依山型', '沿河型', '平原型', '山地型']

village_colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b']


def get_dataset(data_root, crop_size=512):
    train_transform = T.Compose([
        T.Resize(size=crop_size),
        T.ToTensor(),
        T.Normalize(mean=[0.485],
                    std=[0.229]),
    ])
    val_transform = T.Compose([
        T.Resize(size=crop_size),
        T.ToTensor(),
        T.Normalize(mean=[0.485],
                    std=[0.229]),
    ])
    # 分类网络数据集
    train_dst = VillageClss(root_mtvcd=data_root,
                            split="train", transform=train_transform)
    val_dst = VillageClss(root_mtvcd=data_root,
                          split="val", transform=val_transform)
    return train_dst, val_dst


def extract_embeddings(dataloader, model):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    with torch.no_grad():
        model.eval()
        embeddings = np.zeros((len(dataloader.dataset.labels), 2))
        labels = np.zeros(len(dataloader.dataset.labels))
        k = 0
        for images, target in dataloader:
            if torch.cuda.is_available():
                images = torch.from_numpy(np.array(images))
                images = images.to(device)
            embeddings[k:k + len(images)] = model.get_embedding(images).data.cpu().numpy()
            labels[k:k + len(images)] = np.array(target)
            k += len(images)
    return embeddings, labels


def plot_embeddings(embeddings, targets, num_classes, xlim=None, ylim=None):
    plt.figure(figsize=(10, 10))
    for i in range(1, num_classes + 1):
        inds = np.where(targets == i)[0]
        plt.scatter(embeddings[inds, 0], embeddings[inds, 1], alpha=0.5, color=village_colors[i])
    if xlim:
        plt.xlim(xlim[0], xlim[1])
    if ylim:
        plt.ylim(ylim[0], ylim[1])
    plt.legend(village_classes)
    plt.show()


def main():
    model_name = 'siameseNetwork'   # {'classificationNet', 'siameseNetwork','tripletNetwork',
                                    # 'onlinePairSelection', 'onlineTripletSelection'}
    embedding_name = 'embeddingNet'  # {'embeddingNet', 'embeddingResNet'}
    crop_size = 512
    batch_size = 12
    num_classes = 5
    data_root = r'F:\Dataset\tradition_villages1\Classification'
    ckpt = 'checkpoints/mtvc/best_siameseNetwork-valLoss0.2884-Epoch0.pth'
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("Device: %s" % device)
    # set up dataset for contrast learning
    train_dst, test_dst = get_dataset(data_root, crop_size=crop_size)
    train_loader = data.DataLoader(train_dst, batch_size=batch_size, shuffle=True)
    test_loader = data.DataLoader(test_dst, batch_size=batch_size, shuffle=True)
    # set up model
    model_map = {
        'classificationNet': models.classificationNet,
        'siameseNetwork': models.siameseNetwork,
        'tripletNetwork': models.tripletNetwork,
        'onlinePairSelection': models.onlinePairSelection,
        'onlineTripletSelection': models.onlineTripletSelection,
    }
    model = model_map[model_name](embedding_name)
    if ckpt is not None:
        if not os.path.isfile(ckpt):
            raise ValueError('ckpt error!')

        checkpoint = torch.load(ckpt, map_location=torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
        model.load_state_dict(checkpoint["model_state"])
        model.to(device)

        train_embeddings_cl, train_labels_cl = extract_embeddings(train_loader, model)
        plot_embeddings(train_embeddings_cl, train_labels_cl, num_classes)
        val_embeddings_cl, val_labels_cl = extract_embeddings(test_loader, model)
        plot_embeddings(val_embeddings_cl, val_labels_cl, num_classes)


if __name__ == "__main__":
    main()
