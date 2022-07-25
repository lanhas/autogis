import argparse
import yaml

import torch
import numpy as np
import torch.nn.functional as F
from tqdm import tqdm
from torch.utils.data import DataLoader

import datasets
import utils
import models.village_segm as models
from utils import metrics


def main(config):
    val_dataset = datasets.make(config['val_dataset'],
                                **config['val_dataset_args'])
    val_loader = DataLoader(val_dataset, config['batch_size'], pin_memory=True, drop_last=True)
    utils.log('val dataset: {} (x{}), {}'.format(
        val_dataset[0][0].shape, len(val_dataset),
        val_dataset.n_classes))

    model_sv = torch.load(config['load'])
    model = models.load(model_sv)

    utils.log('num params: {}'.format(utils.compute_n_params(model)))

    if config['separable_conv']:
        models.convert_to_separable_conv(model.segmention)
    utils.set_bn_momentum(model.encoder, momentum=0.01)

    val_score = validate(val_loader, model, config['model'], val_dataset.n_classes)

    log_str = 'val loss: {:.4f} | val acc: {:.4f} | val iou:{:.4f}\n'.format(
        val_score['vl'].get_avg(), val_score['va'].get_avg(), val_score['vim'].get_avg())

    classes = ['unknow', 'mountain', 'forest', 'farm', 'water', 'wasteland', 'village']
    log_str += 'classes iou:\n'
    for str in ['   {}:{:.4f}\n'.format(v_class, val_score['vic'].get_avg()[i]) for i, v_class in enumerate(classes)]:
        log_str += str

    utils.log(log_str)


def validate(val_loader, model, model_name, n_classes):
    aves_keys = ['vl', 'va', 'vim', 'vic']
    aves = {k: utils.metrics.MetricTracker() for k in aves_keys}

    model.eval()
    # iterate over data
    for idx, img_data in enumerate(tqdm(val_loader)):
        image, dem, label = img_data[0].cuda(), img_data[1].cuda(), img_data[2].cuda().long()
        with torch.no_grad():
            if model_name[:4] == 'mtss':
                logits = model(image, dem)
            else:
                images = torch.cat((image, dem), 1)
                logits = model(images)

            loss = F.cross_entropy(logits, label)
            acc = utils.compute_dice_coeff(logits, label, n_classes)
            iou_mean, iou_cls = utils.compute_jaccard_index(logits, label, n_classes)

        aves['vl'].add(loss.item())
        aves['va'].add(acc)
        aves['vim'].add(iou_mean)
        aves['vic'].add([*(iou_cls.values())])

    return aves


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='configs/test_villageEP_segm.yaml')

    args = parser.parse_args()

    config = yaml.load(open(args.config, 'r'), Loader=yaml.FullLoader)
    main(config)
