import argparse
import os
import yaml

import torch
import models
import torch.nn as nn
import numpy as np
import utils
from utils import metrics
from torchvision import transforms as T
from torch.utils import data
from utils.visualizer import Visualizer
from datasets.village_clss import Mtvcd, SiameseMtvcd, BalancedBatchSampler, TripletMtvcd
from tqdm import tqdm
from pathlib import Path


def get_dataset(data_root, arch_type='siamese', crop_size=512):
    train_transform = T.Compose([
        T.Resize(size=crop_size),
        T.RandomRotation(180),
        T.ToTensor(),
    ])

    val_transform = T.Compose([
        T.Resize(size=crop_size),
        T.ToTensor(),
        T.Normalize(0.485, 0.229),
    ])
    # siamese网络数据集
    if arch_type == 'classificationNet' or arch_type == 'onlinePairSelection' or arch_type == 'onlineTripletSelection':
        train_dst = Mtvcd(root_mtvcd=data_root,
                          image_set="train", transform=train_transform)
        val_dst = Mtvcd(root_mtvcd=data_root,
                        image_set="val", transform=val_transform)
    elif arch_type == 'siameseNetwork':
        train_dst = SiameseMtvcd(root_mtvcd=data_root,
                                 image_set="train", transform=train_transform)
        val_dst = SiameseMtvcd(root_mtvcd=data_root,
                               image_set="val", transform=val_transform)
    # triplet网络数据集
    elif arch_type == 'tripletNetwork':
        train_dst = TripletMtvcd(root_mtvcd=data_root,
                                 image_set="train", transform=train_transform)
        val_dst = TripletMtvcd(root_mtvcd=data_root,
                               image_set="val", transform=val_transform)
    return train_dst, val_dst


def main(config):
    # 超参数设置
    start_epoch = 1
    config['epochs'] = 75
    ckpt_path = Path('')
    enable_vis = True
    model_name = 'siameseNetwork'  # {'classificationNet', 'siameseNetwork', 'tripletNetwork',
    #  'onlinePairSelection', 'onlineTripletSelection'}
    embedding_name = 'embeddingNet'  # {'embeddingNet', 'embeddingResNet'}
    lr = 1e-4
    n_samples = 2
    num_classes = 6

    batch_size = 6
    step_size = 12
    crop_size = 256
    margin = 1.
    log_interval = 50
    data_root = r'F:\Dataset\tradition_villages_old\Classification'

    # setup visualization
    # visdom
    print_freq = 10
    vis_port = 12370
    vis_env = "main"

    vis = Visualizer(port=vis_port, env=vis_env) if enable_vis else None
    # others

    # setup visualization
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("Device: %s" % device)

    # set up model
    model_map = {
        'classificationNet': models.classificationNet,
        'siameseNetwork': models.siameseNetwork,
        'tripletNetwork': models.tripletNetwork,
        'onlinePairSelection': models.onlinePairSelection,
        'onlineTripletSelection': models.onlineTripletSelection,
    }
    model = model_map[model_name](embedding_name).to(device)

    # set up optimizer
    # optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=weight_decay)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size, gamma=0.1, last_epoch=-1)

    # setup criterion
    if model_name == 'classificationNet':
        criterion = nn.NLLLoss()
    elif model_name == 'siameseNetwork':
        criterion = utils.ContrastiveLoss(margin=margin)
    elif model_name == 'tripletNetwork':
        criterion = utils.TripletLoss(margin=margin)
    elif model_name == 'onlinePairSelection':
        criterion = utils.OnlineContrastiveLoss(margin, utils.HardNegativePairSelector())
    elif model_name == 'onlineTripletSelection':
        criterion = utils.OnlineTripletLoss(margin, utils.RandomNegativeTripletSelector(margin))
    else:
        raise ValueError('model_name error!')
    # setup metrics

    # starting params
    best_loss = 999
    best_acc = 0

    if ckpt_path:
        if ckpt_path.is_file():
            print("=> loading checkpoint '{}'".format(ckpt_path))
            checkpoint = torch.load(ckpt_path)
            if checkpoint['epoch'] > start_epoch:
                start_epoch = checkpoint['epoch']

            best_loss = checkpoint['best_loss']
            best_acc = checkpoint['best_acc']
            model.load_state_dict(checkpoint["state_dict"])
            optimizer.load_state_dict(checkpoint['optimizer'])
            print("=> loaded checkpoint '{}' (epoch {})".format(ckpt_path, checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(ckpt_path))

    # set up dataset for contrast learning
    train_dst, val_dst = get_dataset(data_root, model_name, crop_size=crop_size)
    train_sampler = BalancedBatchSampler(train_dst.labels, num_classes, n_samples)
    val_sampler = BalancedBatchSampler(val_dst.labels, num_classes, n_samples)

    # train_loader = data.DataLoader(train_dst, batch_size=batch_size, shuffle=True)
    # val_loader = data.DataLoader(val_dst, batch_size=batch_size, shuffle=True)

    train_loader = data.DataLoader(train_dst, batch_sampler=train_sampler)
    val_loader = data.DataLoader(val_dst, batch_sampler=val_sampler)
    print("Dataset: Train set: %d, Val set: %d" %
          (len(train_dst), len(val_dst)))
    logger = {'vis': vis, 'print_freq': print_freq}

    # ----------------------------------------train Loop----------------------------------#
    # vis_sample_id = np.random.randint(0, len(val_loader), vis_num_samples,
    #                                     np.int32) if enable_vis else None

    for epoch in range(start_epoch, config['epochs']):
        print('Epoch {}/{}'.format(epoch, config['epochs'] - 1))
        print('-' * 10)

        # run training
        train_score = train(train_loader, model, optimizer, lr_scheduler, criterion, logger, epoch, device)
        valid_score = validate(val_loader, model, criterion, logger, epoch, device)

        is_best_loss = valid_score['valid_loss'] > best_loss

        sv_name = "{}-loss_{:.4f}-Epoch_{}".format(
            model_name, valid_score['valid_loss'], epoch)
        if is_best_loss:
            # print('saving loss best model')
            save_checkpoint({
                'epoch': epoch,
                'arch': model,
                'state_dict': model.state_dict(),
                'best_loss': best_loss,
                'best_acc': best_acc,
                'optimizer': optimizer.state_dict()
            }, is_best_loss, sv_name)
        else:
            # print('saving accuracy best model')
            save_checkpoint({
                'epoch': epoch,
                'arch': model,
                'state_dict': model.state_dict(),
                'best_loss': best_loss,
                'best_acc': best_acc,
                'optimizer': optimizer.state_dict()
            }, is_best_loss, sv_name)


def train(train_loader, model, optimizer, lr_scheduler, criterion, logger, epoch_num, device):
    # train_acc = metrics.MetricTracker()
    train_loss = metrics.MetricTracker()

    meters = {"train_loss": train_loss}

    log_iter = len(train_loader) // logger['print_freq']
    resigual = len(train_loader) % logger['print_freq']
    model.train()
    # iterate over data
    for idx, img_data in enumerate(tqdm(train_loader)):
        meters = make_train_step(img_data, model, optimizer, criterion, meters, device)
        # if clclic_lr
        # lr_scheduler.step()
        if idx and idx % log_iter == 0:
            step = ((epoch_num - 1) * (logger['print_freq'] + resigual)) + (idx / log_iter)
            info = {
                'Train Loss': meters["train_loss"].avg,
            }
            for tag, value in info.items():
                logger['vis'].vis_scalar(tag, step, value)
    # summary
    # if step_lr
    lr_scheduler.step()
    print('Training Loss: {:.4f}'.format(meters["train_loss"].avg))
    print()
    return {'train_loss': meters["train_loss"].avg}


def make_train_step(img_data, model, optimizer, criterion, meters, device):
    images = img_data[0]
    labels = img_data[1] if len(img_data[1]) > 0 else None
    if not type(images) in (tuple, list):
        images = (images,)
    images = tuple(img.to(device, dtype=torch.float32) for img in images)
    labels = torch.from_numpy(np.array(labels).astype("int32")).to(device, dtype=torch.long)

    optimizer.zero_grad()
    outputs = model(*images)

    if type(outputs) not in (tuple, list):
        outputs = (outputs,)

    loss_inputs = outputs
    if labels is not None:
        labels = (labels,)
        loss_inputs += labels
    loss_outputs = criterion(*loss_inputs)
    loss = loss_outputs[0] if type(loss_outputs) in (tuple, list) else loss_outputs
    loss.backward()
    optimizer.step()

    meters["train_loss"].update(loss.item(), outputs[0].shape[0])
    return meters


def validate(valid_loader, model, criterion, logger, epoch_num, device):
    valid_loss = metrics.MetricTracker()

    model.eval()
    # Iterate over Data
    for img_data in tqdm(valid_loader):
        images = img_data[0]
        labels = img_data[1] if len(img_data[1]) > 0 else None
        if not type(images) in (tuple, list):
            images = (images,)
        images = tuple(img.to(device, dtype=torch.float32) for img in images)
        if labels is not None:
            labels = torch.from_numpy(np.array(labels).astype("int32")).to(device, dtype=torch.long)

        outputs = model(*images)
        if type(outputs) not in (tuple, list):
            outputs = (outputs,)
        loss_inputs = outputs
        if labels is not None:
            labels = (labels,)
            loss_inputs += labels
        loss_outputs = criterion(*loss_inputs)
        loss = loss_outputs[0] if type(loss_outputs) in (tuple, list) else loss_outputs
        valid_loss.update(loss.item(), outputs[0].shape[0])
    info = {
        'loss': valid_loss.avg,
    }
    for tag, value in info.items():
        logger['vis'].vis_scalar(tag, epoch_num, value)

    # logging
    print('Validation Loss: {:.4f}'.format(
        valid_loss.avg))
    print()

    return {'valid_loss': valid_loss.avg}


def save_checkpoint(state, is_best, name):
    """ save current model
    """
    checkpoint_dir = Path('./checkpoints/mtvc')
    if not checkpoint_dir.is_dir():
        checkpoint_dir.mkdir()
    filename = checkpoint_dir / (name + '.pth')
    torch.save(state, filename)
    if is_best:
        filename_best = checkpoint_dir / ('best-' + name + '.path')
        filename_best.write_bytes(filename.read_bytes())


if __name__ == "__main__":
    parse = argparse.ArgumentParser()
    parse.add_argument('--config')

    args = parse.parse_args()

    config = yaml.load(open(args.config, 'r'), Loader=yaml.FullLoader)
    main(config)
