import os
import numpy as np
from utils import metrics
from network.unet import UNet, UNetSmall
from torch.utils import data
from datasets.road import RoadSegm
from utils import ext_transforms as et
from utils import losses
from utils.visualizer import Visualizer
from pathlib import Path
from tqdm import tqdm
import torch


# 数据加载
def get_dataset(data_root, crop_size=512):
    train_transform = et.ExtCompose([
            et.ExtRandomScale((0.5, 2.0)),
            et.ExtRandomCrop(size=(crop_size, crop_size), pad_if_needed=True),
            et.ExtRandomHorizontalFlip(),
            et.ExtToTensor(),
            et.ExtNormalize(mean=[0.4100, 0.3831, 0.2886],
                            std=[0.1562, 0.1268, 0.1228],
                            ),
        ])
    val_transform = et.ExtCompose([
            et.ExtRandomCrop(size=(512, 512), pad_if_needed=True),
            et.ExtToTensor(),
            et.ExtNormalize(mean=[0.4075, 0.3807, 0.2848],
                            std=[0.1569, 0.1256, 0.1210],
                            ),
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
    model_name = "unet_small"   # choose model for training (default: unet_small)

    # train
    lr = 1e-3                   # initial learning rate (default: 1e-3)
    weight_decay = 1e-4         #
    jt_loss_weight = 1.0        # weight of Dice or Jaccard term of the joint loss (default: 1.0)
    step_size = 12              # 等间隔调整学习率
    train_batchsize = 6          # mini-batch size (default: 128)
    val_batchsize = 2
    crop_size = 112             # number of cropped pixels from orig image (default: 112)
    lovasz_loss = False
    data_root = Path(r'F:\Dataset\road dataset')  # path to dataset (parent dir of train and val)

    # visdom
    print_freq = 30            # number of time to log per epoch (default: 30)
    vis_port = 12370
    visenv_train = 'train'
    visenv_valid = 'valid'

    # other
    hard_mining = False         # whether use hard negative mining (default: False)
    acc_best = True              # whether store the best model according to validation loss or accuracy (default: True)
    cycle_start_epoch = 30      # start epoch for using cyclic-lr (default: 30)
    vis_num_samples = 2

    # time
    # since = time.time()
    # sv_name = datetime.strftime(datetime.now(), '%Y%m%d_%H%M%S')
    # print('saving file name is ', sv_name)
    # end

    # setup visualization
    vis_train = Visualizer(port=vis_port, env=visenv_train) if enable_vis else None
    vis_valid = Visualizer(port=vis_port, env=visenv_valid) if enable_vis else None

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("Device: %s" % device)

    # setup model
    model_map = {
        'unet': UNet,
        'unet_small': UNetSmall
    }
    model = model_map[model_name]().to(device)

    # set up criterion
    if lovasz_loss:
        print('using lovasz loss function')
        criterion = losses.BCELovaszLoss()
    else:
        criterion = losses.BCEDiceLoss(penalty_weight=jt_loss_weight)

    # setup optimizer
    optimizer = torch.optim.SGD(params=model.parameters(), lr=lr, momentum=0.9, weight_decay=weight_decay)

    # decay LR
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=0.1)

    # starting params
    best_loss = 999
    best_acc = 0

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
        train_dst, batch_size=train_batchsize, shuffle=True, num_workers=2, drop_last=True)
    val_loader = data.DataLoader(
        val_dst, batch_size=val_batchsize, shuffle=True, num_workers=2, drop_last=True)
    print("Dataset: Train set: %d, Val set: %d" %
          (len(train_dst), len(val_dst)))

    train_logger = {'vis': vis_train, 'print_freq': print_freq}
    val_logger = {'vis': vis_valid, 'print_freq': print_freq}

    # ----------------------------------------train Loop---------------------------------- #
    vis_sample_id = np.random.randint(0, len(val_loader), vis_num_samples,
                                      np.int32) if enable_vis else None

    for epoch in range(start_epoch, epochs):
        print('Epoch {}/{}'.format(epoch, epochs - 1))
        print('-' * 10)
        # step the learning rate scheduler
        if epoch == cycle_start_epoch:
            print("Starting cyclic lr")
            print("initial lr: ", lr)
            optimizer = torch.optim.SGD(params=model.parameters(), lr=lr, momentum=0.9, weight_decay=weight_decay)
        if epoch >= cycle_start_epoch:
            lr = cyclic_lr(optimizer, epoch - cycle_start_epoch, init_lr=lr, num_epochs_per_cycle=5,
                           cycle_epochs_decay=2, lr_decay_factor=0.1)
            print("cycling lr: ", lr)
        else:
            lr_scheduler.step()

        # run training and validation
        train_metrics = train(train_loader, model, optimizer, lr_scheduler, criterion, train_logger, epoch, lovasz_loss)
        valid_metrics = validate(val_loader, model, criterion, val_logger, epoch, lovasz_loss)

        is_best_loss = valid_metrics['valid_loss'] < best_loss
        is_best_acc = valid_metrics['valid_acc'] > best_acc

        best_loss = min(valid_metrics['valid_loss'], best_loss)
        best_acc = max(valid_metrics['valid_acc'], best_acc)
        sv_name = "{}-Valloss_{}-Acc_{}-Epoch_{}".format(model_name, \
                  valid_metrics['valid_loss'], valid_metrics['valid_acc'], epoch)
        if acc_best:
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
            }, is_best_acc, sv_name)


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


def cyclic_lr(optimizer, epoch, init_lr=1e-4, num_epochs_per_cycle=5, cycle_epochs_decay=2, lr_decay_factor=0.5):
    epoch_in_cycle = epoch % num_epochs_per_cycle
    lr = init_lr * (lr_decay_factor ** (epoch_in_cycle // cycle_epochs_decay))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    return lr


def make_train_step(data, model, optimizer, criterion, meters, lovasz_loss):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    images = data[0].to(device, dtype=torch.float32)
    labels = data[1].to(device, dtype=torch.float32)
    # zero the parameter gradients
    optimizer.zero_grad()
    outputs = model(images)

    # pay attention to the weighted loss should input logits not probs
    if lovasz_loss:
        loss, BCE_loss, DICE_loss = criterion(outputs, labels)
        outputs = torch.sigmoid(outputs)
    else:
        outputs = torch.sigmoid(outputs)
        loss, BCE_loss, DICE_loss = criterion(outputs, labels)

    # backward
    loss.backward()
    optimizer.step()

    meters["train_acc"].update(metrics.dice_coeff(outputs, labels), outputs.size(0))
    meters["train_loss"].update(loss.item(), outputs.size(0))
    meters["train_IoU"].update(metrics.jaccard_index(outputs, labels), outputs.size(0))
    meters["train_BCE"].update(BCE_loss.item(), outputs.size(0))
    meters["train_DICE"].update(DICE_loss.item(), outputs.size(0))
    meters["outputs"] = outputs
    return meters


def train(train_loader, model, optimizer, lr_scheduler, criterion, logger, epoch_num, lovasz_loss):

    # logging accuracy and loss
    train_acc = metrics.MetricTracker()
    train_loss = metrics.MetricTracker()
    train_IoU = metrics.MetricTracker()
    train_BCE = metrics.MetricTracker()
    train_DICE = metrics.MetricTracker()

    meters = {"train_acc": train_acc, "train_loss": train_loss,
              "train_IoU": train_IoU, "train_BCE": train_BCE,
              "train_DICE": train_DICE, "outputs": None}

    log_iter = len(train_loader) // logger['print_freq']

    model.train()

    lr_scheduler.step()

    # iterate over data
    for idx, data in enumerate(tqdm(train_loader)):
        meters = make_train_step(data, model, optimizer, criterion, meters, lovasz_loss)
        if idx % log_iter == 0:
            step = (epoch_num*logger['print_freq'])+(idx/log_iter)

            # log accurate and loss
            info = {
                'loss': meters["train_loss"].avg,
                'accuracy': meters["train_acc"].avg,
                'IoU': meters["train_IoU"].avg
            }

            for tag, value in info.items():
                logger['vis'].vis_scalar(tag, step, value)

    print('Training Loss: {:.4f} BCE: {:.4f} DICE: {:.4f} Acc: {:.4f} IoU: {:.4f} '.format(
            meters["train_loss"].avg, meters["train_BCE"].avg, meters["train_DICE"].avg, meters["train_acc"].avg, meters["train_IoU"].avg))
    print()
    
    return {'train_loss': meters["train_loss"].avg, 'train_acc': meters["train_acc"].avg,
            'train_IoU': meters["train_IoU"].avg, 'train_BCE': meters["train_BCE"].avg,
            'train_DICE': meters["train_DICE"].avg}


def validate(valid_loader, model, criterion, logger, epoch_num, lovasz_loss):
    # logging accuracy and loss
    valid_acc = metrics.MetricTracker()
    valid_loss = metrics.MetricTracker()
    valid_IoU = metrics.MetricTracker()
    valid_BCE = metrics.MetricTracker()
    valid_DICE = metrics.MetricTracker()

    log_iter = len(valid_loader) // logger['print_freq']

    model.eval()

    # Iterate over Data
    for idx, data in enumerate(tqdm(valid_loader)):
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        images = data[0].to(device, dtype=torch.float32)
        labels = data[1].to(device, dtype=torch.float32)
        # forward
        outputs = model(images)
        # pay attention to the weighted loss should input logits not probs
        if lovasz_loss:
            loss, BCE_loss, DICE_loss = criterion(outputs, labels)
            outputs = torch.sigmoid(outputs)
        else:
            outputs = torch.sigmoid(outputs)
            loss, BCE_loss, DICE_loss = criterion(outputs, labels)
        valid_acc.update(metrics.dice_coeff(outputs, labels), outputs.size(0))
        valid_loss.update(loss.item(), outputs.size(0))
        valid_IoU.update(metrics.jaccard_index(outputs, labels), outputs.size(0))
        valid_BCE.update(BCE_loss.item(), outputs.size(0))
        valid_DICE.update(DICE_loss.item(), outputs.size(0))

        # visdom logging
        if idx % log_iter == 0:
            step = (epoch_num*logger['print_freq'])+(idx/log_iter)

            # log accuracy and loss
            info = {
                'loss': valid_loss.avg,
                'accuracy': valid_acc.avg,
                'IoU': valid_IoU.avg
            }

            for tag, value in info.items():
                logger['vis'].vis_scalar(tag, step, value)

    # logging
    print('Validation Loss: {:.4f} BCE: {:.4f} DICE: {:.4f} Acc: {:.4f} IoU: {:.4f}'.format(
        valid_loss.avg, valid_BCE.avg, valid_DICE.avg, valid_acc.avg, valid_IoU.avg))
    print()

    return {'valid_loss': valid_loss.avg, 'valid_acc': valid_acc.avg,
            'valid_IoU': valid_IoU.avg, 'valid_BCE': valid_BCE.avg,
            'valid_DICE': valid_DICE.avg}


if __name__ == "__main__":
    main()
