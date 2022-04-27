import argparse
import yaml

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm
from pathlib import Path
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

import utils
import datasets
import models.village_clss as models
import utils.village_clss.few_shot as fs
from utils import metrics
from datasets.samplers import CategoriesSampler


def main(config):
    svname = args.name
    if svname is None:
        svname = 'village_clss-classifier_' + config['model_args']['encoder']
        clsfr = config['model_args']['classifier']
        if clsfr != 'linear-classifier':
            svname += '-' + clsfr
    if args.tag is not None:
        svname += '_' + args.tag

    save_path = Path.cwd() / 'save' / svname
    utils.ensure_path(save_path)
    utils.set_log_path(save_path)

    # setup visualize
    writer = SummaryWriter(str(save_path / 'tensorboard'))

    yaml.dump(config, open(save_path / 'config.yaml', 'w'))

    # setupDataset

    # train
    train_dataset = datasets.make(config['train_dataset'],
                                  **config['train_dataset_args'])
    train_loader = DataLoader(train_dataset, config['batch_size'], shuffle=True,
                              num_workers=config['num_workers'], pin_memory=True)

    utils.log('train dataset: {} (x{}), {}'.format(
            torch.cat((train_dataset[0][0], train_dataset[1][0]), dim=0).shape, len(train_dataset),
            train_dataset.n_classes))

    if config.get('visualize_datasets'):
        utils.visualize_dataset(train_dataset, 'train_dataset', writer)

    # setup validate dataset
    if config.get('val_dataset'):
        eval_val = True
        val_dataset = datasets.make(config['val_dataset'],
                                    **config['val_dataset_args'])
        val_loader = DataLoader(val_dataset, config['batch_size'],
                                num_workers=config['num_workers'], pin_memory=True)
        utils.log('val dataset: {} (x{}), {}'.format(
                torch.cat((val_dataset[0][0], val_dataset[1][0]), dim=0).shape, len(val_dataset),
                val_dataset.n_classes))
        if config.get('visualize_datasets'):
            utils.visualize_dataset(val_dataset, 'val_dataset', writer)
    else:
        eval_val = False
        val_loader = None

    # few-shot eval
    if config.get('fs_dataset'):
        eval_fs = True
        ef_epoch = config['eval_fs_epoch']
        if ef_epoch is None:
            ef_epoch = 5

        fs_dataset = datasets.make(config['fs_dataset'],
                                   **config['fs_dataset_args'])
        utils.log('fs dataset: {} (x{}), {}'.format(
            torch.cat((fs_dataset[0][0], fs_dataset[1][0]), dim=0).shape, len(fs_dataset),
            fs_dataset.n_classes))
        if config.get('visualize_datasets'):
            utils.visualize_dataset(fs_dataset, 'fs_dataset', writer)

        eval_fs_args = config['eval_fs_args']
        fs_sampler = CategoriesSampler(
            fs_dataset.label, eval_fs_args['batch_size'],
            eval_fs_args['n_way'],
            eval_fs_args['n_shot'] + eval_fs_args['n_query'],
            ep_per_batch=eval_fs_args['ep_per_batch'])
        fs_loader = DataLoader(fs_dataset, batch_sampler=fs_sampler,
                               num_workers=config['num_workers'], pin_memory=True)
    else:
        eval_fs = False
        fs_loader = None

    # Model and Optimizer

    if config.get('load'):
        model_sv = torch.load(config['load'])
        model = models.load(model_sv)
    else:
        model = models.make(config['model'], **config['model_args'])

    if eval_fs:
        fs_model = models.make('meta-baseline', encoder=None)
        fs_model.encoder = model.encoder
    else:
        fs_model = None

    if config.get('_parallel'):
        model = nn.DataParallel(model)
        if eval_fs:
            fs_model = nn.DataParallel(fs_model)

    utils.log('num params: {}'.format(utils.compute_n_params(model)))

    optimizer, lr_scheduler = utils.make_optimizer(
            model.parameters(),
            config['optimizer'], **config['optimizer_args'])

    max_epoch = config['max_epoch']
    max_va = 0.
    timer_used = utils.Timer()
    timer_epoch = utils.Timer()

    for epoch in range(1, max_epoch + 1):
        print('Epoch {}/{}'.format(epoch, max_epoch))
        print('-' * 10)
        timer_epoch.s()
        train_score = train(train_loader, model, optimizer)
        if lr_scheduler is not None:
            lr_scheduler.step()
        if eval_val:
            val_score = validate(val_loader, model)
        else:
            val_score = None

        if eval_fs and epoch % ef_epoch == 0:
            val_fs_score = validate_fs(fs_loader, fs_model, eval_fs_args)
        else:
            val_fs_score = None

        t_epoch = utils.time_str(timer_epoch.t())
        t_used = utils.time_str(timer_used.t())
        t_estimate = utils.time_str(timer_used.t() / epoch * max_epoch)

        log_str = 'train loss: {:.4f} | train acc: {:.4f}\n'.format(
                  train_score['train_loss'], train_score['train_acc'])
        writer.add_scalars('loss', {'train': train_score['train_loss']}, epoch)
        writer.add_scalars('acc', {'train': train_score['train_acc']}, epoch)

        if eval_val:
            log_str += 'val loss: {:.4f} | val acc: {:.4f}\n'.format(
                        val_score['val_loss'], val_score['val_acc'])
            writer.add_scalars('loss', {'val': val_score['val_loss']}, epoch)
            writer.add_scalars('acc', {'val': val_score['val_acc']}, epoch)

        if eval_fs and epoch % ef_epoch == 0:
            log_str += 'val {}shot acc: {}\n'.format(
                        str(config['eval_fs_args']['n_shot']),
                        val_fs_score['vfs_acc'])
            writer.add_scalars('acc', {'val_fs': val_fs_score['vfs_acc']}, epoch)

        log_str += 'time epoch:{}, time used:{}/{}'.format(t_epoch, t_used, t_estimate)
        utils.log(log_str)

        if config.get('_parallel'):
            model_ = model.module
        else:
            model_ = model

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
            'model_sd': model_.state_dict(),

            'training': training,
        }

        torch.save(save_obj, save_path / 'epoch-last.pth')

        if val_score['val_acc'] > max_va:
            max_va = val_score['val_acc']
            torch.save(save_obj, save_path / 'max-va.pth')

        writer.flush()


def train(train_loader, model, optimizer):
    train_loss = metrics.MetricTracker()
    train_acc = metrics.MetricTracker()

    model.train()
    # iterate over data
    for idx, img_data in enumerate(tqdm(train_loader)):
        data_r = img_data[0].cuda()
        data_d = img_data[1].cuda()
        labels = img_data[2].cuda().long()

        data = torch.cat((data_r, data_d.unsqueeze(1)), dim=1)
        logits = model(data)

        loss = F.cross_entropy(logits, labels)
        acc = utils.compute_acc(logits, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        train_loss.add(loss.item())
        train_acc.add(acc)

    return {'train_loss': train_loss.get_avg(), 'train_acc': train_acc.get_avg()}


def validate(valid_loader, model):
    val_loss = metrics.MetricTracker()
    val_acc = metrics.MetricTracker()

    model.eval()
    # Iterate over Data
    for img_data in tqdm(valid_loader):
        data_r = img_data[0].cuda()
        data_d = img_data[1].cuda()
        labels = img_data[2].cuda().long()

        data = torch.cat((data_r, data_d.unsqueeze(1)), dim=1)
        with torch.no_grad():
            logits = model(data)

            loss = F.cross_entropy(logits, labels)
            acc = utils.compute_acc(logits, labels)

        val_loss.add(loss.item())
        val_acc.add(acc)

    return {'val_loss': val_loss.get_avg(), 'val_acc': val_acc.get_avg()}


def validate_fs(fs_loader, fs_model, fs_args_val):
    vfs_acc = metrics.MetricTracker()

    fs_model.eval()
    np.random.seed(0)
    for data_r, data_d, _ in tqdm(fs_loader,
                        desc='fs-' + str(fs_args_val['n_shot']), leave=False):

        data = torch.cat((data_r, data_d.unsqueeze(1)), dim=1)
        x_shot, x_query = fs.split_shot_query(
            data.cuda(), fs_args_val['n_way'], fs_args_val['n_shot'],
            fs_args_val['n_query'], ep_per_batch=fs_args_val['ep_per_batch'])
        label = fs.make_nk_label(
            fs_args_val['n_way'], fs_args_val['n_query'],
            ep_per_batch=fs_args_val['ep_per_batch']).cuda()
        with torch.no_grad():
            logits = fs_model(x_shot, x_query).view(-1, fs_args_val['n_way'])
            acc = utils.compute_acc(logits, label)
        vfs_acc.add(acc)
    return {'vfs_acc': vfs_acc.get_avg()}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='configs/train_classifier_mul.yaml')
    parser.add_argument('--name', default=None)
    parser.add_argument('--tag', default=None)

    args = parser.parse_args()

    config = yaml.load(open(args.config, 'r'), Loader=yaml.FullLoader)
    main(config)
