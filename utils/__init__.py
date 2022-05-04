import os
import shutil
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import SGD, Adam
from pathlib import Path

_log_path = None


def set_log_path(path):
    global _log_path
    _log_path = path


def del_file(path):
    for file_name in path.iterdir():
        if file_name.is_file():
            file_name.unlink()
        else:
            del_file(file_name)


def ensure_path(path, remove=True):
    path = Path(path)
    if path.exists():
        # if remove and (basename.startswith('_')
        #         or input('{} exists, remove? ([y]/n): '.format(path)) != 'n'):
        if remove:
            del_file(path)
    else:
        path.mkdir()


def log(obj, filename='log.txt'):
    print(obj)
    if _log_path is not None:
        with open(os.path.join(_log_path, filename), 'a') as f:
            print(obj, file=f)


class Timer():

    def __init__(self):
        self.v = time.time()

    def s(self):
        self.v = time.time()

    def t(self):
        return time.time() - self.v


def set_gpu(gpu):
    print('set gpu:', gpu)
    os.environ['CUDA_VISIBLE_DEVICES'] = gpu


def time_str(t):
    if t >= 3600:
        return '{:.1f}h'.format(t / 3600)
    if t >= 60:
        return '{:.1f}m'.format(t / 60)
    return '{:.1f}s'.format(t)


def compute_logits(feat, proto, metric='dot', temp=1.0):
    assert feat.dim() == proto.dim()

    if feat.dim() == 2:
        if metric == 'dot':
            logits = torch.mm(feat, proto.t())
        elif metric == 'cos':
            logits = torch.mm(F.normalize(feat, dim=-1),
                              F.normalize(proto, dim=-1).t())
        elif metric == 'sqr':
            logits = -(feat.unsqueeze(1) -
                       proto.unsqueeze(0)).pow(2).sum(dim=-1)

    elif feat.dim() == 3:
        if metric == 'dot':
            logits = torch.bmm(feat, proto.permute(0, 2, 1))
        elif metric == 'cos':
            logits = torch.bmm(F.normalize(feat, dim=-1),
                               F.normalize(proto, dim=-1).permute(0, 2, 1))
        elif metric == 'sqr':
            logits = -(feat.unsqueeze(2) -
                       proto.unsqueeze(1)).pow(2).sum(dim=-1)

    return logits * temp


def compute_acc(logits, label, reduction='mean'):
    ret = (torch.argmax(logits, dim=1) == label).float()
    if reduction == 'none':
        return ret.detach()
    elif reduction == 'mean':
        return ret.mean().item()


def compute_fast_hist(logits, label, n_classes):
    confusion_matrix = torch.zeros((n_classes, n_classes)).cuda()

    label_pred = torch.argmax(logits, dim=1)
    for label_true, label_pred in zip(label, label_pred):
        mask = (label_true >= 0) & (label_true < n_classes)
        hist = torch.bincount(
            n_classes * label_true[mask] + label_pred[mask],
            minlength=n_classes ** 2,
        ).reshape(n_classes, n_classes)
        confusion_matrix += hist
    return confusion_matrix


def compute_jaccard_index(logits, label, n_classes):
    hist = compute_fast_hist(logits, label, n_classes)
    iou = torch.diag(hist) / (hist.sum(axis=1) + hist.sum(axis=0) - torch.diag(hist))
    iou_mean = iou[iou < float('inf')].mean()
    iou_cls = dict(zip(range(n_classes), iou.cpu().detach().numpy()))
    return iou_mean.item(), iou_cls


def compute_dice_coeff(logits, label, n_classes):
    hist = compute_fast_hist(logits, label, n_classes)
    acc = torch.diag(hist).sum() / hist.sum()
    acc_cls = torch.diag(hist) / hist.sum(axis=1)
    acc_mean = acc_cls[acc_cls < float('inf')].mean()
    # acc_cls = dict(zip(range(n_classes), acc_cls.cpu().detach().numpy()))
    return acc_mean.item()

# https://stackoverflow.com/questions/48260415/pytorch-how-to-compute-iou-jaccard-index-for-semantic-segmentation
# https://github.com/ternaus/robot-surgery-segmentation/blob/master/evaluate.py
def bin_jaccard_index(input, target):
    """IoU calculation """
    num_in_target = input.size(0)

    pred = input.view(num_in_target, -1)
    truth = target.view(num_in_target, -1)

    # intersection = (pred*truth).long().sum(1).data.cpu()[0]
    intersection = (pred * truth).sum(1)

    # union = input.long().sum().data.cpu()[0] + target.long().sum().data.cpu()[0] - intersection
    union = pred.sum(1) + truth.sum(1) - intersection

    score = (intersection + 1e-15) / (union + 1e-15)

    return score.mean().item()


# https://github.com/pytorch/pytorch/issues/1249
def bin_dice_coeff(input, target):
    num_in_target = input.size(0)

    smooth = 1.

    pred = input.view(num_in_target, -1)
    truth = target.view(num_in_target, -1)

    intersection = (pred * truth).sum(1)

    loss = (2. * intersection + smooth) /(pred.sum(1) + truth.sum(1) + smooth)

    return loss.mean().item()

def compute_n_params(model, return_str=True):
    tot = 0
    for p in model.parameters():
        w = 1
        for x in p.shape:
            w *= x
        tot += w
    if return_str:
        if tot >= 1e6:
            return '{:.1f}M'.format(tot / 1e6)
        else:
            return '{:.1f}K'.format(tot / 1e3)
    else:
        return tot


def make_optimizer(params, name, lr, weight_decay=None, milestones=None):
    if weight_decay is None:
        weight_decay = 0.
    if name == 'sgd':
        optimizer = SGD(params, lr, momentum=0.9, weight_decay=weight_decay)
    elif name == 'adam':
        optimizer = Adam(params, lr, weight_decay=weight_decay)
    else:
        raise ValueError('optimizer name error! please check')
    if milestones:
        lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones)
    else:
        lr_scheduler = None
    return optimizer, lr_scheduler


def visualize_dataset(dataset, name, writer, n_samples=16):
    demo = []
    for i in np.random.choice(len(dataset), n_samples):
        demo.append(dataset.convert_raw(dataset[i][0]))
    writer.add_images('visualize_' + name, torch.stack(demo))
    writer.flush()


def freeze_bn(model):
    for m in model.modules():
        if isinstance(m, nn.BatchNorm2d):
            m.eval()


def set_bn_momentum(model, momentum=0.1):
    for m in model.modules():
        if isinstance(m, nn.BatchNorm2d):
            m.momentum = momentum


def fix_bn(model):
    for m in model.modules():
        if isinstance(m, nn.BatchNorm2d):
            m.eval()


