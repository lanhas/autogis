import argparse
import yaml

import torch
import torch.nn.functional as F
from tqdm import tqdm
from pathlib import Path
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

import datasets
import utils
import models.road_segm as models
from utils import metrics


def main(config):
    svname = args.name
    if svname is None:
        svname = 'road_segm-' + config['model']
    if args.tag is not None:
        svname += '_' + args.tag

    save_path = Path.cwd() / 'save' / svname
    utils.ensure_path(save_path)
    utils.set_log_path(save_path)

    # setup visualize
    writer = SummaryWriter(str(save_path / 'tensorboard'))
    yaml.dump(config, open(save_path / 'config.yaml', 'w'))

    # setup datasets
    train_dataset = datasets.make(config['train_dataset'],
                                  **config['train_dataset_args'])
    val_dataset = datasets.make(config['val_dataset'],
                                **config['val_dataset_args'])

    train_loader = DataLoader(train_dataset, config['batch_size'], shuffle=True, pin_memory=True, drop_last=True)
    val_loader = DataLoader(val_dataset, config['batch_size'], pin_memory=True, drop_last=True)

    utils.log('train dataset: {} (x{})'.format(
        train_dataset[0][0].shape, len(train_dataset)))
    utils.log('val dataset: {} (x{})'.format(
        val_dataset[0][0].shape, len(train_dataset)))

    if config.get('load'):
        model_sv = torch.load(config['load'])
        model = models.load(model_sv)
    else:
        model = models.make(config['model'])

    utils.log('num params: {}'.format(utils.compute_n_params(model)))

    optimizer, lr_scheduler = utils.make_optimizer(
        model.parameters(),
        config['optimizer'], **config['optimizer_args'])

    # setup others
    max_epoch = config['max_epoch']
    max_vi = 0.

    timer_used = utils.Timer()
    timer_epoch = utils.Timer()

    for epoch in range(1, max_epoch + 1):
        print('Epoch {}/{}'.format(epoch, max_epoch))
        print('-' * 10)
        timer_epoch.s()
        # run training
        writer.add_scalar('lr', optimizer.param_groups[0]['lr'], epoch)
        train_score = train(train_loader, model, optimizer)
        val_score = validate(val_loader, model)

        # post
        if lr_scheduler is not None:
            lr_scheduler.step()

        t_epoch = utils.time_str(timer_epoch.t())
        t_used = utils.time_str(timer_used.t())
        t_estimate = utils.time_str(timer_used.t() / epoch * max_epoch)

        log_str = 'train loss: {:.4f} | train acc: {:.4f} | train iou:{:.4f}\n'.format(
            train_score['tl'].get_avg(), train_score['ta'].get_avg(), train_score['tim'].get_avg())
        log_str += 'val loss: {:.4f} | val acc: {:.4f} | val iou:{:.4f}\n'.format(
            val_score['vl'].get_avg(), val_score['va'].get_avg(), val_score['vim'].get_avg())

        writer.add_scalars('loss', {'train': train_score['tl'].get_avg()}, epoch)
        writer.add_scalars('loss', {'val': val_score['vl'].get_avg()}, epoch)

        writer.add_scalars('acc', {'train': train_score['tl'].get_avg()}, epoch)
        writer.add_scalars('acc', {'val': val_score['va'].get_avg()}, epoch)

        writer.add_scalars('mean_IoU', {'train': train_score['tim'].get_avg()}, epoch)
        writer.add_scalars('mean_IoU', {'val': val_score['vim'].get_avg()}, epoch)

        log_str += 't_epoch:{} | t_used:{}/{}'.format(t_epoch, t_used, t_estimate)
        utils.log(log_str)

        training = {
            'epoch': epoch,
            'optimizer': config['optimizer'],
            'optimizer_args': config['optimizer_args'],
            'optimizer_sd': optimizer.state_dict(),
        }
        save_obj = {
            'file': __file__,
            'config': config,

            'model': config['model'],
            'model_args': config['model_args'],
            'model_sd': model.state_dict(),

            'training': training,
        }

        torch.save(save_obj, save_path / 'epoch-last.pth')
        if val_score['vim'].get_avg() > max_vi:
            max_vi = val_score['vim'].get_avg()
            torch.save(save_obj, save_path / 'max-vi.pth')
        writer.flush()


def train(train_loader, model, optimizer):
    aves_keys = ['tl', 'ta', 'tim']
    aves = {k: utils.metrics.MetricTracker() for k in aves_keys}

    model.train()
    # iterate over data
    for idx, img_data in enumerate(tqdm(train_loader)):
        image, label = img_data[0].cuda(), img_data[1].cuda().float()
        outputs = model(image).squeeze(dim=1)

        loss = F.binary_cross_entropy(outputs, label)
        acc_mean = utils.bin_dice_coeff(outputs, label)
        iou_mean = utils.bin_jaccard_index(outputs, label)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        aves['tl'].add(loss.item())
        aves['ta'].add(acc_mean)
        aves['tim'].add(iou_mean)

    return aves


def validate(val_loader, model):
    aves_keys = ['vl', 'va', 'vim']
    aves = {k: utils.metrics.MetricTracker() for k in aves_keys}

    model.eval()
    # iterate over data
    for idx, img_data in enumerate(tqdm(val_loader)):
        image, label = img_data[0].cuda(), img_data[1].cuda().float()
        with torch.no_grad():
            outputs = model(image).squeeze(dim=1)

            loss = F.binary_cross_entropy(outputs, label)
            acc_mean = utils.bin_dice_coeff(outputs, label)
            iou_mean = utils.bin_jaccard_index(outputs, label)

        aves['vl'].add(loss.item())
        aves['va'].add(acc_mean)
        aves['vim'].add(iou_mean)

    return aves


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='configs/train_road.yaml')
    parser.add_argument('--name', default=None)
    parser.add_argument('--tag', default=None)

    args = parser.parse_args()

    config = yaml.load(open(args.config, 'r'), Loader=yaml.FullLoader)
    main(config)
