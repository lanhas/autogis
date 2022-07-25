import argparse

import yaml

import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import scipy.stats
from tqdm import tqdm
from pathlib import Path
from PIL import Image
from torch.utils.data import DataLoader
from datasets.villageEP import village_cmap

import datasets
import models.village_clss as models
import utils
import utils.metrics as metrics
import matplotlib.pyplot as plt


def mean_confidence_interval(data, confidence=0.95):
    a = 1.0 * np.array(data)
    n = len(a)
    se = scipy.stats.sem(a)
    h = se * scipy.stats.t.ppf((1 + confidence) / 2., n - 1)
    return h


def main(config):
    # dataset
    dataset = datasets.make(config['dataset'], **config['dataset_args'])
    utils.log('dataset: {} (x{}), {}'.format(
            dataset[0][0].shape, len(dataset), dataset.n_classes))

    loader = DataLoader(dataset)

    # model
    if config.get('load') is None:
        model = models.make('classifier', encoder=None)
    else:
        model = models.load(torch.load(config['load']))

    if config.get('load_encoder') is not None:
        encoder = models.load(torch.load(config['load_encoder'])).encoder
        model.encoder = encoder

    if config.get('_parallel'):
        model = nn.DataParallel(model)

    model.eval()
    utils.log('num params: {}'.format(utils.compute_n_params(model)))

    # validate
    aves, va_lst = validate(model, loader)
    # draw_cam
    # draw_cam(model, config['data_path'], visual_heatmap=False)

    print('acc={:.2f} +- {:.2f} (%), loss={:.4f}'.format(
                aves['va'].get_avg() * 100,
                mean_confidence_interval(va_lst) * 100,
                aves['vl'].get_avg()))


def validate(model, loader):
    # testing
    aves_keys = ['vl', 'va']
    aves = {k: metrics.MetricTracker() for k in aves_keys}

    np.random.seed(0)
    va_lst = []
    for img_data in tqdm(loader):
        with torch.no_grad():
            images = img_data[0].cuda()
            labels = img_data[1].cuda().long()
            logits = model(images)

            loss = F.cross_entropy(logits, labels)
            acc = utils.compute_acc(logits, labels)

            aves['vl'].add(loss.item())
            aves['va'].add(acc)
            va_lst.append(acc)
    return aves, va_lst


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='../configs/test_villageSP_clss.yaml')
    parser.add_argument('--sauc', action='store_true')

    args = parser.parse_args()

    config = yaml.load(open(args.config, 'r'), Loader=yaml.FullLoader)

    main(config)

