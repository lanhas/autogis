import network
import torch
import os
import torch.nn as nn
import utils
import numpy as np
from torchvision import transforms as T
from torch.utils import data
from torch.optim import lr_scheduler
from datasets.mtvcd import Mtvcd, SiameseMtvcd, BalancedBatchSampler, TripletMtvcd
from tqdm import tqdm
from visdom import Visdom


def get_dataset(data_root, arch_type='siamese', crop_size=512, cropval=512):
    train_transform = T.Compose([
        T.Resize(size=crop_size),
        # T.RandomScale((0.5, 2.0)),
        T.RandomRotation(180),
        T.ToTensor(),
        T.Normalize((0.485), (0.229)),
    ])

    val_transform = T.Compose([
        T.Resize(size=cropval),
        T.ToTensor(),
        T.Normalize((0.485), (0.229)),
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


def train_epoch(train_loader, model, loss_fn, optimizer, device, log_interval, metrics=[]):
    for metric in metrics:
        metric.reset()
    model.train()
    losses = []
    total_loss = 0

    for batch_idx, (images, target) in tqdm(enumerate(train_loader)):
        target = target if len(target) > 0 else None
        if not type(images) in (tuple, list):
            images = (images,)

        images = tuple(img.to(device, dtype=torch.float32) for img in images)
        if target is not None:
            target = torch.from_numpy(np.array(target).astype("int32")).to(device, dtype=torch.long)

        optimizer.zero_grad()
        outputs = model(*images)

        if type(outputs) not in (tuple, list):
            outputs = (outputs,)

        loss_inputs = outputs
        if target is not None:
            target = (target,)
            loss_inputs += target

        loss_outputs = loss_fn(*loss_inputs)
        loss = loss_outputs[0] if type(loss_outputs) in (tuple, list) else loss_outputs
        losses.append(loss.item())
        total_loss += loss.item()
        loss.backward()
        optimizer.step()

        for metric in metrics:
            metric(outputs, target, loss_outputs)

        # if batch_idx % log_interval == 0:
        #     message = '\nTrain: [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
        #         batch_idx * len(images[0]), len(train_loader.dataset),
        #         100. * batch_idx / len(train_loader), np.mean(losses))
        #     for metric in metrics:
        #         message += '\t{}: {}'.format(metric.name(), metric.value())
        #     print(message)
        #     losses = []
    total_loss /= (batch_idx + 1)
    return total_loss, metrics


def test_epoch(val_loader, model, loss_fn, device, metrics=[]):
    with torch.no_grad():
        for metric in metrics:
            metric.reset()
        model.eval()
        val_loss = 0
        for batch_idx, (images, target) in tqdm(enumerate(val_loader)):
            target = target if len(target) > 0 else None
            if not type(images) in (tuple, list):
                images = (images,)
            images = tuple(img.to(device, dtype=torch.float32) for img in images)
            if target is not None:
                target = torch.from_numpy(np.array(target).astype("int32")).to(device, dtype=torch.long)

            outputs = model(*images)
            if type(outputs) not in (tuple, list):
                outputs = (outputs,)
            loss_inputs = outputs
            if target is not None:
                target = (target,)
                loss_inputs += target
            loss_outputs = loss_fn(*loss_inputs)
            loss = loss_outputs[0] if type(loss_outputs) in (tuple, list) else loss_outputs
            val_loss += loss.item()

            for metric in metrics:
                metric(outputs, target, loss_outputs)
    return val_loss, metrics


def train():
    ckpt = None
    enable_vis = False
    continue_training = False
    # train
    model_name = 'siameseNetwork'  # {'classificationNet', 'siameseNetwork', 'tripletNetwork',
    #  'onlinePairSelection', 'onlineTripletSelection'}
    embedding_name = 'embeddingNet'  # {'embeddingNet', 'embeddingResNet'}
    n_samples = 3
    weight_decay = 1e-4
    num_classes = 6
    n_epochs = 50
    step_size = 10
    crop_size = 512
    margin = 1.
    log_interval = 50

    # setup visualization
    metrics = []
    # visdom
    vis_port = 12370
    vis_env = "main"
    name = ['train_loss', 'valid_loss']
    vis = Visdom(port=vis_port, env=vis_env) if enable_vis else None
    # others
    data_root = r'F:\Dataset\tradition_villages1\Classification'
    # setup visualization
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("Device: %s" % device)

    # set up dataset for contrast learning
    train_dst, val_dst = get_dataset(data_root, model_name, crop_size=crop_size)
    train_sampler = BalancedBatchSampler(train_dst.labels, num_classes, n_samples)
    val_sampler = BalancedBatchSampler(val_dst.labels, num_classes, n_samples)
    if model_name == 'classificationNet':
        lr = 1e-2
        batch_size = 12
        loss_fn = nn.NLLLoss()
        train_loader = data.DataLoader(train_dst, batch_size=batch_size, shuffle=True)
        val_loader = data.DataLoader(val_dst, batch_size=batch_size, shuffle=True)
    elif model_name == 'siameseNetwork':
        lr = 1e-3
        batch_size = 6
        loss_fn = utils.ContrastiveLoss(margin=margin)
        train_loader = data.DataLoader(train_dst, batch_size=batch_size, shuffle=True)
        val_loader = data.DataLoader(val_dst, batch_size=batch_size, shuffle=True)
    elif model_name == 'tripletNetwork':
        lr = 1e-3
        batch_size = 6
        loss_fn = utils.TripletLoss(margin=margin)
        train_loader = data.DataLoader(train_dst, batch_size=batch_size, shuffle=True)
        val_loader = data.DataLoader(val_dst, batch_size=batch_size, shuffle=True)
    elif model_name == 'onlinePairSelection':
        lr = 1e-3
        train_loader = data.DataLoader(train_dst, batch_sampler=train_sampler)
        val_loader = data.DataLoader(val_dst, batch_sampler=val_sampler)
        loss_fn = utils.OnlineContrastiveLoss(margin, utils.HardNegativePairSelector())
    elif model_name == 'onlineTripletSelection':
        lr = 1e-3
        train_loader = data.DataLoader(train_dst, batch_sampler=train_sampler)
        val_loader = data.DataLoader(val_dst, batch_sampler=val_sampler)
        loss_fn = utils.OnlineTripletLoss(margin, utils.RandomNegativeTripletSelector(margin))
    else:
        raise ValueError('model_name error!')

    train_sampler = BalancedBatchSampler(train_dst.labels, num_classes, n_samples)
    val_sampler = BalancedBatchSampler(val_dst.labels, num_classes, n_samples)
    train_loader = data.DataLoader(train_dst, batch_sampler=train_sampler)
    val_loader = data.DataLoader(val_dst, batch_sampler=val_sampler)
    print("Dataset: Train set: %d, Val set: %d" %
          (len(train_dst), len(val_dst)))
    # set up model
    model_map = {
        'classificationNet': network.classificationNet,
        'siameseNetwork': network.siameseNetwork,
        'tripletNetwork': network.tripletNetwork,
        'onlinePairSelection': network.onlinePairSelection,
        'onlineTripletSelection': network.onlineTripletSelection,
    }
    # set up optimizer
    model = model_map[model_name](embedding_name)
    optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=weight_decay)
    scheduler = lr_scheduler.StepLR(optimizer, step_size, gamma=0.1, last_epoch=-1)

    def save_ckpt(path):
        """ save current model
        """
        torch.save({
            "cur_itrs": cur_itrs,
            "model_state": model.module.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "scheduler_state": scheduler.state_dict(),
            "best_score": best_score,
        }, path)
        print("Model saved as %s" % path)

    print()
    utils.mkdir('checkpoints/mtvc')
    # Restore
    best_score = 0.0
    cur_itrs = 0
    best_loss = float('inf')
    if ckpt is not None and model_name.split('_')[0] == 'siamese':
        if not os.path.isfile(ckpt):
            raise ValueError('ckpt error!')

        checkpoint = torch.load(ckpt, map_location=torch.device('cpu'))
        model.load_state_dict(checkpoint["model_state"])
        model = nn.DataParallel(model)
        model.to(device)
        if continue_training:
            optimizer.load_state_dict(checkpoint["optimizer_state"])
            scheduler.load_state_dict(checkpoint["scheduler_state"])
            cur_itrs = checkpoint["cur_itrs"]
            best_score = checkpoint['best_score']
            print("Training state restored from %s" % ckpt)
        print("Model restored from %s" % ckpt)
        del checkpoint  # free memory
    else:
        print("[!] Retrain")
        model = nn.DataParallel(model)
        model.to(device)

    # ----------------------------------------train Loop----------------------------------#
    # vis_sample_id = np.random.randint(0, len(val_loader), vis_num_samples,
    #                                     np.int32) if enable_vis else None

    for epoch in range(n_epochs):
        train_loss, metrics = train_epoch(train_loader, model, loss_fn, optimizer, device, log_interval, metrics)
        scheduler.step()

        message = 'Epoch:{}/{}.Train set: Average loss: {:.4f}\n'.format(epoch + 1, n_epochs, train_loss)
        for metric in metrics:
            message += '\t{}: {:.4f}\n'.format(metric.name(), metric.value())
        val_loss, metrics = test_epoch(val_loader, model, loss_fn, device, metrics)
        val_loss /= len(val_loader)
        if val_loss < best_loss:
            best_loss = val_loss
            save_ckpt('checkpoints/mtvc/best_%s-valLoss%.4f-Epoch%d.pth' %
                      (model_name, val_loss, epoch))
        message += 'Epoch:{}/{}. Valid set: Average loss: {:.4f}\n'.format(epoch + 1, n_epochs, val_loss)
        for metric in metrics:
            message += '\t{}: {:.4}\n'.format(metric.name(), metric.value())
        print(message)
        # visdom
        if enable_vis:
            vis.line(np.column_stack((train_loss, val_loss)), [epoch], win='train_log', update='append',
                     opts=dict(title='losses', legend=name))


if __name__ == "__main__":
    train()
