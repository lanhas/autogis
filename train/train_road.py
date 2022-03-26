import os
import torch
import numpy as np
import torch.nn as nn
import network
from copy import deepcopy
from utils import metrics
from torch.utils import data
from datasets.road import RoadSegm
from utils import ext_transforms as et
from utils.visualizer import Visualizer
from pathlib import Path
from tqdm import tqdm


# 数据加载
def get_dataset(data_root, crop_size=256):
    train_transform = et.ExtCompose([
            # et.ExtRandomScale((0.5, 2.0)),
            et.ExtRandomCrop(size=(crop_size, crop_size), pad_if_needed=True),
            et.ExtRandomHorizontalFlip(),
            et.ExtToTensor(),
        ])
    val_transform = et.ExtCompose([
            et.ExtRandomCrop(size=(crop_size, crop_size), pad_if_needed=True),
            et.ExtToTensor(),
        ])
    train_dst = RoadSegm(root_dir=data_root, status='train', transform=train_transform)
    val_dst = RoadSegm(root_dir=data_root, status='valid', transform=val_transform)
    return train_dst, val_dst


def main():
    # 超参数设置
    start_epoch = 0
    epochs = 75                 # number of total epochs to run (default: 75)
    ckpt_path = None            # path to latest checkpoint (default: none)
    enable_vis = True
    hard_mining = True
    model_name = "dense-unet"   # choose model for training (default: unet_small)

    # train
    # lr_policy = 'cyclic_lr'
    lr_policy = 'step_lr'
    lr = 5e-4                   # initial learning rate (default: 1e-3)
    max_lr = 2e-3
    weight_decay = 1e-4         # weight decay of SGD optimizer,
    bce_loss_weight = 1.0       # weight of Dice or Jaccard term of the joint loss (default: 1.0)
    step_size = 12              # 等间隔调整学习率
    train_batch_size = 2         # mini-batch size (default: 128)
    valid_batch_size = 2
    crop_size = 256             # number of cropped pixels from orig image (default: 112)
    data_root = Path(r'F:\Dataset\road dataset')  # path to dataset (parent dir of train and val)

    # visdom
    print_freq = 300            # number of time to log per epoch (default: 2)
    vis_port = 12370
    vis_env = 'mean'

    # other
    hard_mining = False         # whether use hard negative mining (default: False)
    vis_num_samples = 2

    # setup visualization
    vis = Visualizer(port=vis_port, env=vis_env) if enable_vis else None

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("Device: %s" % device)

    # setup model
    model_map = {
        'unet': network.unet,
        'unet_small': network.unet_small,
        'dense_unet': network.dense_unet,
    }
    model = model_map[model_name]().to(device)

    # set up criterion
    criterion = nn.BCELoss()

    # setup optimizer
    optimizer = torch.optim.SGD(params=model.parameters(), lr=lr, momentum=0.9, weight_decay=weight_decay)
    # optimizer = torch.optim.Adam(params=model.parameters(), lr=lr)

    # decay LR
    if lr_policy == 'cyclic_lr':
        lr_scheduler = torch.optim.lr_scheduler.CyclicLR(optimizer, lr, max_lr, )
    elif lr_policy == 'step_lr':
        lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=0.1)
    else:
        raise ValueError('lr_policy error! Please check!')

    # starting params
    best_loss = 999
    best_acc = 0
    best_iou = 0

    if ckpt_path:
        if os.path.isfile(ckpt_path):
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

    # set up dataloader
    train_dst, val_dst = get_dataset(data_root, crop_size)
    train_loader = data.DataLoader(
        train_dst, batch_size=train_batch_size, shuffle=True, num_workers=2, drop_last=True)
    val_loader = data.DataLoader(
        val_dst, batch_size=valid_batch_size, shuffle=True, num_workers=2, drop_last=True)
    print("Dataset: Train set: %d, Val set: %d" %
          (len(train_dst), len(val_dst)))

    logger = {'vis': vis, 'print_freq': print_freq}

    # ----------------------------------------train Loop---------------------------------- #
    vis_sample_id = np.random.randint(0, len(val_loader), vis_num_samples,
                                      np.int32) if enable_vis else None

    for epoch in range(start_epoch, epochs):
        print('Epoch {}/{}'.format(epoch, epochs - 1))
        print('-' * 10)

        # run training and validation
        train_score = train(train_loader, model, optimizer, lr_scheduler, criterion, logger, hard_mining, epoch)
        valid_score = validate(val_loader, model, criterion, logger, epoch)

        is_best_iou = valid_score['valid_IoU'] > best_iou
        # is_best_loss = valid_score['valid_loss'] < best_loss
        # is_best_acc = valid_score['valid_acc'] > best_acc

        best_loss = min(valid_score['valid_loss'], best_loss)
        best_acc = max(valid_score['valid_acc'], best_acc)
        best_iou = max(valid_score['valid_IoU'], best_iou)

        sv_name = "{}-loss_{:.4f}-Acc_{:.4f}-IoU_{:.4f}-Epoch_{}".format(
            model_name, valid_score['valid_loss'], valid_score['valid_loss'], valid_score['valid_IoU'], epoch)
        if is_best_iou:
            # print('saving loss best model')
            save_checkpoint({
                'epoch': epoch,
                'arch': model,
                'state_dict': model.state_dict(),
                'best_loss': best_loss,
                'best_acc': best_acc,
                'optimizer': optimizer.state_dict()
            }, is_best_iou, sv_name)
        else:
            # print('saving accuracy best model')
            save_checkpoint({
                'epoch': epoch,
                'arch': model,
                'state_dict': model.state_dict(),
                'best_loss': best_loss,
                'best_acc': best_acc,
                'optimizer': optimizer.state_dict()
            }, is_best_iou, sv_name)


def save_checkpoint(state, is_best, name):
    """ save current model
    """
    checkpoint_dir = Path('./checkpoints/road')
    if not checkpoint_dir.is_dir():
        checkpoint_dir.mkdir()
    filename = checkpoint_dir / (name + '.pth')
    torch.save(state, filename)
    if is_best:
        filename_best = checkpoint_dir / ('best' + name + '.pth')
        filename_best.write_bytes(filename.read_bytes())


def make_train_step(img_data, model, optimizer, criterion, meters):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    images = img_data[0].to(device, dtype=torch.float32)
    labels = img_data[1].to(device, dtype=torch.float32)
    # zero the parameter gradients
    optimizer.zero_grad()
    outputs = model(images)

    outputs = torch.sigmoid(outputs)
    loss = criterion(outputs, labels)

    # backward
    loss.backward()
    optimizer.step()

    meters["train_acc"].update(metrics.dice_coeff(outputs, labels), outputs.size(0))
    meters["train_loss"].update(loss.item(), outputs.size(0))
    meters["train_IoU"].update(metrics.jaccard_index(outputs, labels), outputs.size(0))
    meters["outputs"] = outputs
    return meters


def train(train_loader, model, optimizer, lr_scheduler, criterion, logger, hard_mining, epoch_num):

    # logging accuracy and loss
    train_acc = metrics.MetricTracker()
    train_loss = metrics.MetricTracker()
    train_iou = metrics.MetricTracker()

    meters = {"train_acc": train_acc, "train_loss": train_loss,
              "train_IoU": train_iou,  "outputs": None}

    log_iter = len(train_loader) // logger['print_freq']
    resigual = len(train_loader) % logger['print_freq']
    model.train()

    cache = None
    cached_loss = 0

    # iterate over data
    for idx, img_data in enumerate(tqdm(train_loader)):
        meters = make_train_step(img_data, model, optimizer, criterion, meters)
        # lr_scheduler.step()

        # hard negative mining
        if hard_mining:
            if cache is None or cached_loss < meters['train_loss'].val:
                cached_loss = meters['train_loss'].val
                cache = deepcopy(img_data)
            if idx % 50 == 0 and cache is not None:
                meters = make_train_step(cache, model, optimizer, criterion, meters)
                cache = None
                cached_loss = 0

        if idx % log_iter == 0:
            step = (epoch_num*(logger['print_freq'] + resigual))+idx

            # log accurate and loss
            info = {
                'Train loss': meters["train_loss"].avg,
                'Train accuracy': meters["train_acc"].avg,
                'Train IoU': meters["train_IoU"].avg
            }

            for tag, value in info.items():
                logger['vis'].vis_scalar(tag, step, value)
    # if step_lr
    lr_scheduler.step()
    print('Training Loss: {:.4f} Acc: {:.4f} IoU: {:.4f} '.format(
            meters["train_loss"].avg, meters["train_acc"].avg, meters["train_IoU"].avg))
    print()
    
    return {'train_loss': meters["train_loss"].avg, 'train_acc': meters["train_acc"].avg,
            'train_IoU': meters["train_IoU"].avg}


def validate(valid_loader, model, criterion, logger, epoch_num):
    # logging accuracy and loss
    valid_acc = metrics.MetricTracker()
    valid_loss = metrics.MetricTracker()
    valid_iou = metrics.MetricTracker()

    log_iter = len(valid_loader) // logger['print_freq']
    residual = (len(valid_loader) % logger['print_freq']) // log_iter

    model.eval()
    # Iterate over Data
    for idx, img_data in enumerate(tqdm(valid_loader)):
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        images = img_data[0].to(device, dtype=torch.float32)
        labels = img_data[1].to(device, dtype=torch.float32)
        # forward
        outputs = model(images)

        outputs = torch.sigmoid(outputs)
        loss = criterion(outputs, labels)
        valid_acc.update(metrics.dice_coeff(outputs, labels), outputs.size(0))
        valid_loss.update(loss.item(), outputs.size(0))
        valid_iou.update(metrics.jaccard_index(outputs, labels), outputs.size(0))

        # visdom logging
        if idx and idx % log_iter == 0:
            step = (epoch_num*(logger['print_freq']+residual))+(idx/log_iter)

            # log accuracy and loss
            info = {
                'loss': valid_loss.avg,
                'accuracy': valid_acc.avg,
                'IoU': valid_iou.avg
            }

            for tag, value in info.items():
                logger['vis'].vis_scalar(tag, step, value)

    # logging
    print('Validation Loss: {:.4f} Acc: {:.4f} IoU: {:.4f}'.format(
        valid_loss.avg, valid_acc.avg, valid_iou.avg))
    print()

    return {'valid_loss': valid_loss.avg, 'valid_acc': valid_acc.avg,
            'valid_IoU': valid_iou.avg}


if __name__ == "__main__":
    main()
