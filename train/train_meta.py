import argparse
import yaml

import torch
import torch.nn as nn
import torch.nn.functional as F
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
        svname = 'village_clss-{}-{}shot'.format(
                   config['model'], config['train_fs_args']['n_shot']) \
                   + '-' + config['model_args']['encoder']
    if args.tag is not None:
        svname += '_' + args.tag

    save_path = Path.cwd() / 'save' / svname
    utils.ensure_path(save_path)
    utils.set_log_path(save_path)

    # setup visualize
    writer = SummaryWriter(save_path / 'tensorboard')

    yaml.dump(config, open(save_path / 'config.yaml', 'w'))

    # setup Dataset

    # train
    train_dataset = datasets.make(config['train_dataset'],
                                  **config['train_dataset_args'])
    utils.log('train dataset: {} (x{}), {}'.format(
        train_dataset[0][0].shape, len(train_dataset),
        train_dataset.n_classes))
    if config.get('visualize_datasets'):
        utils.visualize_dataset(train_dataset, 'train_dataset', writer)

    train_fs_args = config['train_fs_args']
    train_sampler = CategoriesSampler(
        train_dataset.label, train_fs_args['batch_size'],
        train_fs_args['n_way'],
        train_fs_args['n_shot'] + train_fs_args['n_query'],
        ep_per_batch=train_fs_args['ep_per_batch'])

    train_loader = DataLoader(train_dataset, batch_sampler=train_sampler,
                              num_workers=config['num_workers'], pin_memory=True)

    # val
    val_dataset = datasets.make(config['val_dataset'],
                                **config['val_dataset_args'])
    utils.log('val dataset: {} (x{}), {}'.format(
        val_dataset[0][0].shape, len(val_dataset),
        val_dataset.n_classes))
    if config.get('visualize_datasets'):
        utils.visualize_dataset(val_dataset, 'val_dataset', writer)

    val_fs_args = config['val_fs_args']
    val_sampler = CategoriesSampler(
        val_dataset.label, val_fs_args['batch_size'],
        val_fs_args['n_way'],
        val_fs_args['n_shot'] + val_fs_args['n_query'],
        ep_per_batch=val_fs_args['ep_per_batch'])
    val_loader = DataLoader(val_dataset, batch_sampler=val_sampler,
                            num_workers=config['num_workers'], pin_memory=True)

    # tval
    if config.get('tval_dataset'):
        eval_tval = True
        tval_dataset = datasets.make(config['tval_dataset'],
                                     **config['tval_dataset_args'])
        utils.log('tval dataset: {} (x{}), {}'.format(
            tval_dataset[0][0].shape, len(tval_dataset),
            tval_dataset.n_classes))
        if config.get('visualize_datasets'):
            utils.visualize_dataset(tval_dataset, 'tval_dataset', writer)
        tval_fs_args = config['tval_fs_args']
        tval_sampler = CategoriesSampler(
            tval_dataset.label, tval_fs_args['batch_size'],
            tval_fs_args['n_way'],
            tval_fs_args['n_shot'] + tval_fs_args['n_query'],
            ep_per_batch=tval_fs_args['ep_per_batch'])
        tval_loader = DataLoader(tval_dataset, batch_sampler=tval_sampler,
                                 num_workers=config['num_workers'], pin_memory=True)
    else:
        eval_tval = False
        tval_loader = None

    # setup Model and optimizer
    if config.get('load'):
        model_sv = torch.load(config['load'])
        model = models.load(model_sv)
    else:
        model = models.make(config['model'], **config['model_args'])

        if not config.get('load_encoder'):
            encoder = models.load(torch.load(config['load_encoder'])).encoder
            model.encoder.load_state_dict(encoder.state_dict())

    if config.get('_parallel'):
        model = nn.DataParallel(model)

    utils.log('num params: {}'.format(utils.compute_n_params(model)))

    optimizer, lr_scheduler = utils.make_optimizer(
        model.parameters(),
        config['optimizer'], **config['optimizer_args'])

    ########

    max_epoch = config['max_epoch']
    max_va = 0.
    timer_used = utils.Timer()
    timer_epoch = utils.Timer()

    for epoch in range(1, max_epoch + 1):
        print('Epoch {}/{}'.format(epoch, max_epoch))
        print('-' * 10)

        timer_epoch.s()
        train_score = train(train_loader, model, optimizer, train_fs_args)
        val_score = validate(val_loader, model, val_fs_args)
        if eval_tval:
            tval_score = validate(tval_loader, model, tval_fs_args)
        else:
            tval_score = None

        t_epoch = utils.time_str(timer_epoch.t())
        t_used = utils.time_str(timer_used.t())
        t_estimate = utils.time_str(timer_used.t() / epoch * max_epoch)

        log_str = 'train loss: {:.4f} | train acc: {:.4f}\n'.format(
                   train_score['train_loss'], train_score['train_acc'])

        writer.add_scalars('loss', {'train': train_score['train_loss']}, epoch)
        writer.add_scalars('acc', {'train': train_score['train_acc']}, epoch)

        log_str += 'val loss: {:.4f} | val acc: {:.4f}\n'.format(
                    val_score['val_loss'], val_score['val_acc'])
        writer.add_scalars('loss', {'val': val_score['val_loss']}, epoch)
        writer.add_scalars('acc', {'val': val_score['val_acc']}, epoch)

        if eval_tval:
            log_str += 'tval loss: {:.4f} | tval acc: {:.4f}\n'.format(
                        tval_score['val_loss'], tval_score['val_acc'])
            writer.add_scalars('loss', {'val': tval_score['val_loss']}, epoch)
            writer.add_scalars('acc', {'val': tval_score['val_acc']}, epoch)

        log_str += 'time epoch: {}, time used: {}/{}'.format(t_epoch, t_used, t_estimate)
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


def train(train_loader, model, optimizer, fs_args_train):
    train_loss = metrics.MetricTracker()
    train_acc = metrics.MetricTracker()

    model.train()
    # iterate over data
    for idx, img_data in enumerate(tqdm(train_loader)):
        x_shot, x_query = fs.split_shot_query(img_data[0].cuda(),
                                              fs_args_train['n_way'], fs_args_train['n_shot'],
                                              fs_args_train['n_query'],
                                              ep_per_batch=fs_args_train['ep_per_batch'])
        label = fs.make_nk_label(fs_args_train['n_way'], fs_args_train['n_query'],
                                 ep_per_batch=fs_args_train['ep_per_batch']).cuda().long()

        logits = model(x_shot, x_query).view(-1, fs_args_train['n_way'])
        loss = F.cross_entropy(logits, label)
        acc = utils.compute_acc(logits, label)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        train_loss.add(loss.item())
        train_acc.add(acc)

    return {'train_loss': train_loss.get_avg(), 'train_acc': train_acc.get_avg()}


def validate(valid_loader, model, fs_args_val):
    val_loss = metrics.MetricTracker()
    val_acc = metrics.MetricTracker()

    model.eval()
    # Iterate over Data
    for img_data in tqdm(valid_loader):
        x_shot, x_query = fs.split_shot_query(img_data[0].cuda(),
                                              fs_args_val['n_way'], fs_args_val['n_shot'], fs_args_val['n_query'],
                                              ep_per_batch=fs_args_val['ep_per_batch'])
        label = fs.make_nk_label(fs_args_val['n_way'], fs_args_val['n_query'],
                                 ep_per_batch=fs_args_val['ep_per_batch']).cuda().long()

        with torch.no_grad():
            logits = model(x_shot, x_query).view(-1, fs_args_val['n_way'])
            loss = F.cross_entropy(logits, label)
            acc = utils.compute_acc(logits, label)

        val_loss.add(loss.item())
        val_acc.add(acc)

    return {'val_loss': val_loss.get_avg(), 'val_acc': val_acc.get_avg()}


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='configs/train_meta.yaml')
    parser.add_argument('--name', default=None)
    parser.add_argument('--tag', default=None)
    args = parser.parse_args()

    config = yaml.load(open(args.config, 'r'), Loader=yaml.FullLoader)

    main(config)
