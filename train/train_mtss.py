import utils
import numpy as np
import network
import torch
import torch.nn as nn
from tqdm import tqdm
from pathlib import Path
from torch.utils import data
from utils.metrics import MtssMetrics
from datasets.mtsd import villageFactorsSegm
from utils import mul_transforms as et
from utils.visualizer import Visualizer


# 数据加载
def get_dataset(data_root, crop_size=512):
    train_transform = et.ExtCompose([
            et.ExtRandomScale((0.5, 2.0)),
            et.ExtRandomCrop(size=(crop_size, crop_size), pad_if_needed=True),
            et.ExtRandomHorizontalFlip(),
            et.ExtToTensor(),
            et.ExtNormalize(mean_image=[0.2737, 0.3910, 0.3276],
                            std_image=[0.1801, 0.1560, 0.1301],
                            mean_dem=[0.4153],
                            std_dem=[0.2405]
                            ),
        ])
    val_transform = et.ExtCompose([
            et.ExtRandomCrop(size=(crop_size, crop_size), pad_if_needed=True),
            et.ExtToTensor(),
            et.ExtNormalize(mean_image=[0.2737, 0.3910, 0.3276],
                            std_image=[0.1801, 0.1560, 0.1301],
                            mean_dem=[0.4153],
                            std_dem=[0.2405]),
        ])
    train_dst = villageFactorsSegm(root_mtsd=data_root,
                                   image_set='train', transform=train_transform)
    val_dst = villageFactorsSegm(root_mtsd=data_root, 
                                 image_set='val', transform=val_transform)
    return train_dst, val_dst


def main():
    # 超参数设置
    start_epoch = 0
    epochs = 75
    ckpt_path = Path('')
    enable_vis = True
    model_name = 'mtss_resnet50'

    num_classes = 7

    # deeplab options
    separable_conv = False
    output_stride = 16  # choices=[8, 16]

    # train
    lr = 1e-4
    lr_policy = 'cyclic_lr'
    max_lr = 1e-3
    weight_decay = 1e-4
    step_size = 12

    train_batch_size = 2
    valid_batch_size = 2
    crop_size = 256
    loss_type = 'cross_entropy'  # choices=['cross_entropy', 'focal_loss']
    data_root = Path(r'F:\Dataset\tradition_villages1\Segmentation')

    # visdom
    print_freq = 10
    vis_port = 12370
    vis_env = 'main'

    # other
    vis_num_samples = 2

    # setup visualization
    vis = Visualizer(port=vis_port, env=vis_env) if enable_vis else None

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("Device: %s" % device)

    # setup model
    model_map = {
        'mtss_resnet50': network.mtss_resnet50,
        'deeplabv3plus_resnet50': network.deeplabv3plus_resnet50,
        'mtss_resnet101': network.mtss_resnet101,
        'deeplabv3plus_resnet101': network.deeplabv3plus_resnet101,
        'mtss_mobilenet': network.mtss_mobilenet,
        'deeplabv3plus_mobilenet': network.deeplabv3plus_mobilenet
    }
    model = model_map[model_name](num_classes=num_classes, output_stride=output_stride,
                                  pretrained_backbone=True).to(device)
    if separable_conv:
        network.convert_to_separable_conv(model.segmention)
    utils.set_bn_momentum(model.backbone, momentum=0.01)

    # setup criterion
    if loss_type == 'focal_loss':
        criterion = utils.FocalLoss(ignore_index=255, size_average=True)
    elif loss_type == 'cross_entropy':
        criterion = nn.CrossEntropyLoss(ignore_index=255, reduction='mean')
    else:
        raise ValueError("loss_type name error! Please check!")

    # set up optimizer
    optimizer = torch.optim.SGD(params=model.parameters(), lr=lr, momentum=0.9, weight_decay=weight_decay)

    # decay LR
    if lr_policy == 'poly':
        lr_scheduler = utils.PolyLR(optimizer, epochs, power=0.9)
    elif lr_policy == 'step':
        lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=0.1)
    elif lr_policy == 'cyclic_lr':
        lr_scheduler = torch.optim.lr_scheduler.CyclicLR(optimizer, base_lr=lr,
                                                         max_lr=max_lr, step_size_up=50, step_size_down=50)
    else:
        raise ValueError('lr name error!please check!')

    # set up metrics
    metrics = MtssMetrics(num_classes)

    # starting params
    best_loss = 999
    best_acc = 0
    best_iou = 0

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

    # set up dataloader
    # load data
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

        # run training
        train_score = train(train_loader, model, model_name, optimizer, lr_scheduler, criterion, metrics, logger, epoch)
        val_score = validate(val_loader, model, model_name, criterion, metrics, logger, epoch)

        # is_best_loss = val_score['Loss'] < best_loss
        # is_best_acc = val_score['Overall Acc'] > best_acc
        is_best_iou = val_score['Mean IoU'] > best_iou

        best_loss = min(val_score['Loss'], best_loss)
        best_acc = max(val_score['Overall Acc'], best_acc)
        best_iou - max(val_score['Mean IoU'], best_iou)
        sv_name = "{}-loss_{:.4f}-Acc_{:.4f}-IoU_{:.4f}-Epoch_{}".format(
                    model_name, val_score['Loss'], val_score['Overall Acc'], val_score['Mean IoU'], epoch)
        if is_best_iou:
            # print('saving loss best model')
            save_checkpoint({
                'epoch': epoch,
                'arch': model,
                'state_dict': model.state_dict(),
                'best_loss': best_loss,
                'best_acc': best_acc,
                'best_iou': best_iou,
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
                'best_iou': best_iou,
                'optimizer': optimizer.state_dict()
            }, is_best_iou, sv_name)


def train(train_loader, model, model_name, optimizer, lr_scheduler, criterion, metrics, logger, epoch_num):
    # reset scheduler
    metrics.reset()
    log_iter = len(train_loader) // logger['print_freq']
    resigual = len(train_loader) % logger['print_freq']
    model.train()
    # iterate over data
    for idx, img_data in enumerate(tqdm(train_loader)):
        train_metrics = make_train_step(img_data, model, model_name, optimizer, criterion, metrics)
        # if cyclic_lr
        lr_scheduler.step()
        if idx and idx % log_iter == 0:
            step = (epoch_num * (logger['print_freq']+resigual)) + (idx / log_iter)
            score = train_metrics.get_results()
            # log accurate and loss
            info = {
                'Train Loss': score["Loss"],
                'Train Overall Acc': score["Overall Acc"],
                'Train Class IoU': score["Class IoU"],
                'Train Mean IoU': score['Mean IoU']
            }
            for tag, value in info.items():
                logger['vis'].vis_scalar(tag, step, value)
    # summary
    score = metrics.get_results()
    # if step_lr
    # lr_scheduler.step()
    print('Training Loss: {:.4f} Overall Acc: {:.4f} Mean IoU: {:.4f} '.format(
        score["Loss"], score["Overall Acc"], score['Mean IoU']))
    print()
    return score


def make_train_step(img_data, model, model_name, optimizer, criterion, metrics):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    images = img_data[0].to(device, dtype=torch.float32)
    dems = img_data[1].to(device, dtype=torch.float32)
    labels = img_data[2].to(device, dtype=torch.long)
    # zero the parameter gradients
    optimizer.zero_grad()
    if model_name[:4] == 'mtss':
        outputs = model(images, dems)
    else:
        images = torch.cat((images, dems), 1)
        outputs = model(images)

    loss = criterion(outputs, labels)

    # backward
    loss.backward()
    optimizer.step()

    preds = outputs.detach().max(dim=1)[1].cpu().numpy()
    targets = labels.cpu().numpy()
    metrics.update(targets, preds, loss)
    return metrics


def validate(valid_loader, model, model_name, criterion, meters, logger, epoch_num):
    
    log_iter = len(valid_loader) // logger['print_freq']
    residual = (len(valid_loader) % logger['print_freq']) // log_iter
    # reset metrics
    meters.reset()
    
    model.eval()
    # Iterate over Data
    for idx, img_data in enumerate(tqdm(valid_loader)):
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        images = img_data[0].to(device, dtype=torch.float32)
        dems = img_data[1].to(device, dtype=torch.float32)
        labels = img_data[2].to(device, dtype=torch.long)
        # forward
        if model_name[:4] == 'mtss':
            outputs = model(images, dems)
        else:
            images = torch.cat((images, dems), 1)
            outputs = model(images)
        # pay attention to the weighted loss should input logits not probs
        loss = criterion(outputs, labels)

        preds = outputs.detach().max(dim=1)[1].cpu().numpy()
        targets = labels.cpu().numpy()
        meters.update(targets, preds, loss)

        # visdom logging

        if idx and idx % log_iter == 0:
            step = (epoch_num * (logger['print_freq']+residual)) + (idx / log_iter)
            score = meters.get_results()
            # log accuracy and loss
            info = {
                'Valid Loss': score["Loss"],
                'Valid Overall Acc': score["Overall Acc"],
                'Valid Class IoU': score["Class IoU"],
                'Valid Mean IoU': score['Mean IoU']
            }

            for tag, value in info.items():
                logger['vis'].vis_scalar(tag, step, value)

    # summary
    score = meters.get_results()
    # logging
    print('Valid Loss: {:.4f} Overall Acc: {:.4f} Mean IoU: {:.4f} '.format(
        score["Loss"], score["Overall Acc"], score['Mean IoU']))
    print()
    return score


def save_checkpoint(state, is_best, name):
    """ save current model
    """
    checkpoint_dir = Path('./checkpoints/mtss')
    if not checkpoint_dir.is_dir():
        checkpoint_dir.mkdir()
    filename = checkpoint_dir / (name + '.pth')
    torch.save(state, filename)
    if is_best:
        filename_best = checkpoint_dir / ('best-' + name + '.pth')
        filename_best.write_bytes(filename.read_bytes())


if __name__ == '__main__':
    main()
