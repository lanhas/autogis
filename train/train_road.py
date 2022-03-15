from pathlib import Path
import sys
sys.path.append(r'F:\code\python\SemanticSegmentation\autogis')
from tqdm import tqdm
import utils
import os
import random
import numpy as np
import datetime
from network.unet import UNet, UNetSmall
from torch.utils import data
from datasets.road import RoadSegm
from utils import ext_transforms as et 
from utils.metrics import MtssMetrics, RoadMetrics
from utils import losses
from utils.visualizer import Visualizer
import torch
import torch.nn as nn


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
            et.ExtToTensor(),
            et.ExtNormalize(mean=[0.4075, 0.3807, 0.2848],
                            std=[0.1569, 0.1256, 0.1210],
                            ),
        ])
    train_dst = RoadSegm(root_dir=data_root,
                                    status='train', transform=train_transform)
    val_dst = RoadSegm(root_dir=data_root, 
                                    status='valid', transform=val_transform)
    return train_dst, val_dst

def main():
    #--------------------------------#
    #-----------超参数----------------#
    #--------------------------------#
    ckpt = None
    enable_vis = True
    model_name = "unet_small"

    # train
    lr = 1e-3
    weight_decay = 1e-4
    jt_loss_weight = 1.0
    total_itrs = 5000  
    step_size = 800       # 等间隔调整学习率
    epochs = 75
    batch_size = 4
    val_batch_size = 2
    crop_size = 512
    continue_training = False
    lovasz_loss = False
    lr_policy = 'poly'

    # visdom
    vis_port = 12370
    vis_env = 'main'

    # other
    data_root = Path.cwd() / "datasets/rood"
    val_interval = 500
    training_log = True
    vis_num_samples = 2
    random_seed = 1
    
    # ------------------------------#
    #--------------end--------------#
    #-------------------------------#

    # setup visualization
    vis = Visualizer(port=vis_port, env=vis_env) if enable_vis else None
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("Device: %s" %device)

    # setup random seed
    torch.manual_seed(random_seed)
    np.random.seed(random_seed)
    random.seed(random_seed)

    # setup training_log
    if training_log == True:
        file_handle = open('./train/logs/trainLog_villageLand.txt', mode='a+')
        file_handle.writelines([
        '*---------------------------------------------------------------------------------------------------------------------------------*\n',
        '*--------------------------------------------------------- 训练日志 ---------------------------------------------------------------*\n',
        '训练日期: ' + str(datetime.datetime.strftime(datetime.datetime.now(),'%Y-%m-%d %H:%M:%S')) +'\n',
            ])
        file_handle.close()
    train_dst, val_dst = get_dataset(data_root, crop_size)
    train_loader = data.DataLoader(
        train_dst, batch_size=batch_size, shuffle=True, num_workers=2, drop_last=True)
    val_loader = data.DataLoader(
        val_dst, batch_size=val_batch_size, shuffle=True, num_workers=2, drop_last=True)
    print("Dataset: Train set: %d, Val set: %d" %
          (len(train_dst), len(val_dst)))

    # setup model
    model_map = {
        'unet': UNet,
        'unet_small': UNetSmall
    }
    model = model_map[model_name]()
    # metrics = RoadMetrics()
    metrics = MtssMetrics(2)

    # setup optimizer
    optimizer = torch.optim.SGD(params=model.parameters(), lr=lr, momentum=0.9, weight_decay=weight_decay)
    if lr_policy=='poly':
        scheduler = utils.PolyLR(optimizer, total_itrs, power=0.9)
    elif lr_policy=='step':
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=0.1)
    # set up apex
    # set up criterion
    if lovasz_loss:
        print('using lovasz loss function')
        criterion = losses.BCELovaszLoss()
    else:
        criterion = losses.BCEDiceLoss(penalty_weight=jt_loss_weight)
    def save_ckpt(path):
        """ save current model
        """
        torch.save({
            "cur_iters": cur_iters,
            "model_state": model.module.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "scheduler_state": scheduler.state_dict(),
            "best_score": best_score,
        }, path)
        print("Model saved as %s" % path)
    
    utils.mkdir('checkpoints/Road')
    # Restore
    best_score = 0.0
    cur_iters = 0
    cur_epochs = 0
    if ckpt is not None and os.path.isfile(ckpt):
        checkpoint = torch.load(ckpt, map_location=torch.device('cpu'))
        model.load_state_dict(checkpoint["model_state"])
        model = nn.DataParallel(model)
        model.to(device)
        if continue_training:
            optimizer.load_state_dict(checkpoint["optimizer_state"])
            scheduler.load_state_dict(checkpoint["scheduler_state"])
            cur_iters = checkpoint["cur_iters"]
            best_score = checkpoint['best_score']
            print("Training state restored from %s" % ckpt)
        print("Model restored from %s" % ckpt)
        del checkpoint  # free memory
    else:
        print("[!] Retrain")
        model = nn.DataParallel(model)
        model.to(device)

    #----------------------------------------train Loop----------------------------------#
    vis_sample_id = np.random.randint(0, len(val_loader), vis_num_samples,
                                        np.int32) if enable_vis else None

    interval_loss = 0
    while True: # cur_iters < total_itrs:
        model.train()
        cur_epochs += 1
        for (images, labels) in train_loader:
            cur_iters += 1

            images = images.to(device, dtype=torch.float32)
            labels = labels.to(device, dtype=torch.float32)

            optimizer.zero_grad()
            outputs = model(images)

            # pay attention to the weighted loss should input logits not probs
            if lovasz_loss:
                loss, BCE_loss, DICE_loss = criterion(outputs, labels)
                outputs = torch.sigmoid(outputs)
            else:
                outputs = torch.sigmoid(outputs)
                loss, BCE_loss, DICE_loss = criterion(outputs, labels)

            loss.backward()
            optimizer.step()

            np_loss = loss.detach().cpu().numpy()
            interval_loss += np_loss
            if vis is not None:
                vis.vis_scalar('Loss', cur_iters, np_loss)
             
            if (cur_iters) % 10 == 0:
                interval_loss = interval_loss / 10
                print("Epoch %d, Itrs %d/%d, Loss=%.4f" %
                      (cur_epochs, cur_iters, total_itrs, interval_loss))
                interval_loss = 0.0
                
            if (cur_iters) % val_interval == 0:
                print("validation...")
                model.eval()
                val_score, ret_samples = validate(
                       model=model, loader=val_loader, device=device, metrics=metrics, ret_samples_ids=vis_sample_id)
                print(metrics.to_str(val_score))

                if val_score['Mean IoU'] > best_score:  # save best model
                    best_score = val_score['Mean IoU']
                    save_ckpt('checkpoints/road/best_%s-Score%.4f-Epoch%d-Iters%d-Total_Loss%.4f.pth' %
                          (model_name, best_score, cur_epochs, cur_iters, interval_loss))
                # else:
                #     save_ckpt('checkpoints/villageLand/latest_%s-Score%.4f-Epoch%d-Iters%d-Total_Loss%.4f.pth' %
                #           (model_name, cur_score, cur_epochs, cur_iters, interval_loss))

                if vis is not None:  # visualize validation score and samples
                    vis.vis_scalar("[Val] Overall Acc", cur_iters, val_score['Overall Acc'])
                    vis.vis_scalar("[Val] Mean IoU", cur_iters, val_score['Mean IoU'])
                    vis.vis_table("[Val] Class IoU", val_score['Class IoU'])
                if training_log == True:
                    file_handle = open('./train/logs/trainLog_Road.txt', mode='a+')
                    file_handle.writelines([
                    '*------------------------------------------------*\n'
                    'The prediction of model %s\n' %(model_name),
                    'Epoch: %d\n' %(cur_epochs),
                    'Itrs %d/%d\n' %(cur_iters, total_itrs),
                    'Total_Loss_%f\n' %(interval_loss),
                    metrics.to_str(val_score)
                    ])
                    file_handle.close()
                model.train()
            scheduler.step()
            if cur_iters >=  total_itrs:
                return

def make_train_step(data, model, optimizer, criterion, metrics, device, lovasz_loss):
    
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

    loss.backward()
    optimizer.step()

    metrics["train_acc"].update(metrics.dice_coeff(outputs, labels), outputs.size(0))
    metrics["train_loss"].update(loss.data[0], outputs.size(0))
    metrics["train_IoU"].update(metrics.jaccard_index(outputs, labels), outputs.size(0))
    metrics["train_BCE"].update(BCE_loss.data[0], outputs.size(0))
    metrics["train_DICE"].update(DICE_loss.data[0], outputs.size(0))
    metrics["outputs"] = outputs
    return metrics

def train(model, loader, optimizer, scheduler, criterion, metrics, device, lovasz_loss):
    metrics.reset()
    model.train()
    scheduler.step()
    for idx, (images, labels) in enumerate(loader):
        metrics = make_train_step((images, labels), model, optimizer, criterion, metrics, lovasz_loss, device)

    print('Training Loss: {:.4f} BCE: {:.4f} DICE: {:.4f} Acc: {:.4f} IoU: {:.4f} '.format(
            metrics["train_loss"].avg, metrics["train_BCE"].avg, metrics["train_DICE"].avg, metrics["train_acc"].avg, metrics["train_IoU"].avg))
    print()
    
    return {'train_loss': metrics["train_loss"].avg, 'train_acc': metrics["train_acc"].avg, 
            'train_IoU': metrics["train_IoU"].avg, 'train_BCE': metrics["train_BCE"].avg,
            'train_DICE': metrics["train_DICE"].avg}
            
def validate(model, loader, device, metrics, ret_samples_ids=None):
    metrics.reset()
    ret_samples = []
    with torch.no_grad():
        for i, (images, labels) in tqdm(enumerate(loader)):
            images = images.to(device, dtype=torch.float32)
            labels = labels.to(device, dtype=torch.long)
            outputs = model(images)
            preds = outputs.detach().max(dim=1)[1].cpu().numpy()
            targets = labels.cpu().numpy()

            metrics.update(targets, preds)
            if ret_samples_ids is not None and i in ret_samples_ids:
                ret_samples.append(
                    (images[0].detach().cpu().numpy(), targets[0], preds[0])
                )
        score = metrics.get_results()
    return score, ret_samples

if __name__ == "__main__":
    train()