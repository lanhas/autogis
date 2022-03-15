import utils
import os
import random
import numpy as np
import network
import datetime
from torch.utils import data
from datasets.mtsd import villageFactorsSegm
from utils import mul_transforms as et
from utils.metrics import MtssMetrics
import torch
import torch.nn as nn
from tqdm import tqdm
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


def validate(model_name, model, loader, device, metrics, ret_samples_ids=None):
    metrics.reset()
    ret_samples = []
    with torch.no_grad():
        for i, (images, dems, labels) in tqdm(enumerate(loader)):
            images = images.to(device, dtype=torch.float32)
            
            labels = labels.to(device, dtype=torch.long)
            if model_name[:4] == 'mtss':
                dems = dems.to(device, dtype=torch.float32)
                outputs = model(images, dems)
            else:
                dems = dems.to(device, dtype=torch.float32)
                images = torch.cat((images, dems), 1)
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


def train():

    # 超参数设置
    ckpt = None
    enable_vis = False
    # model_name = 'mtss_resnet50'
    model_name = 'deeplabv3plus_resnet50'

    # train
    lr = 1e-2
    weight_decay = 1e-4     # SGD优化器权值衰减
    total_itrs = 30000
    step_size = 10000       # 等间隔调整学习率
    num_classes = 7
    batch_size = 2
    crop_val = False
    val_batch_size = 2
    crop_size = 512
    continue_training = False
    loss_type = 'cross_entropy'     # choices=['cross_entropy', 'focal_loss']
    lr_policy = 'poly'              # choice=['poly', 'step']
    
    # deeplab options
    separable_conv = False
    output_stride = 16      # choices=[8, 16]
    # visdom
    vis_port = 12370
    vis_env = 'main'

    # others
    data_root = r'F:\Dataset\tradition_villages1\Segmentation'
    val_interval = 100  # 每n个iter计算一次acc、iou
    training_log = True
    random_seed = 1
    vis_num_samples = 2
    enable_apex = False

    # end
    # setup visualization
    vis = Visualizer(port=vis_port, env=vis_env, use_incoming_socket=False) if enable_vis else None
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("Device: %s" % device)

    # setup random seed
    torch.manual_seed(random_seed)
    np.random.seed(random_seed)
    random.seed(random_seed)

    # setup dataloader
    if not crop_val:
        val_batch_size = 1
    
    # setup training_log
    # utils.mkdir('./train/logs')
    if training_log:
        file_handle = open('train/logs/trainLog_villageLand.txt', mode='a+')
        file_handle.writelines([
        '*---------------------------------------------------------------------------------------------------------------------------------*\n',
        '*--------------------------------------------------------- 训练日志 ---------------------------------------------------------------*\n',
        '训练日期： ' + str(datetime.datetime.strftime(datetime.datetime.now(), '%Y-%m-%d %H:%M:%S')) +'\n',
            ])
        file_handle.close()

    # load data
    train_dst, val_dst = get_dataset(data_root, crop_size)
    train_loader = data.DataLoader(
        train_dst, batch_size=batch_size, shuffle=True, num_workers=2, drop_last=True)
    val_loader = data.DataLoader(
        val_dst, batch_size=val_batch_size, shuffle=True, num_workers=2, drop_last=True)
    print("Dataset: Train set: %d, Val set: %d" %
          (len(train_dst), len(val_dst)))

    # set up model
    model_map = {
        'mtss_resnet50': network.mtss_resnet50,
        'deeplabv3plus_resnet50': network.deeplabv3plus_resnet50,
        'mtss_resnet101': network.mtss_resnet101,
        'deeplabv3plus_resnet101': network.deeplabv3plus_resnet101,
        'mtss_mobilenet': network.mtss_mobilenet,
        'deeplabv3plus_mobilenet': network.deeplabv3plus_mobilenet
    }
    model = model_map[model_name](num_classes=num_classes, output_stride=output_stride, pretrained_backbone=True)
    if separable_conv:
        network.convert_to_separable_conv(model.segmention)
    utils.set_bn_momentum(model.backbone, momentum=0.01)
    metrics = MtssMetrics(num_classes)

    # set up optimizer
    optimizer = torch.optim.SGD(params=model.parameters(), lr=lr, momentum=0.9, weight_decay=weight_decay)
    if lr_policy == 'poly':
        scheduler = utils.PolyLR(optimizer, total_itrs, power=0.9)
    elif lr_policy == 'step':
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=0.1)
    # set up apex
    # set up criterion
    if loss_type == 'focal_loss':
        criterion = utils.FocalLoss(ignore_index=255, size_average=True)
    elif loss_type == 'cross_entropy':
        criterion = nn.CrossEntropyLoss(ignore_index=255, reduction='mean')

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
    
    utils.mkdir('checkpoints/villageLand')
    # Restore
    best_score = 0.0
    cur_itrs = 0
    cur_epochs = 0
    if ckpt is not None and os.path.isfile(ckpt):
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

    #----------------------------------------train Loop----------------------------------#
    vis_sample_id = np.random.randint(0, len(val_loader), vis_num_samples,
                                        np.int32) if enable_vis else None

    interval_loss = 0
    while True: # cur_itrs < total_itrs:
        model.train()
        cur_epochs += 1
        for (images, dems, labels) in train_loader:
            cur_itrs += 1

            images = images.to(device, dtype=torch.float32)
            dems = dems.to(device, dtype=torch.float32)
            labels = labels.to(device, dtype=torch.long)

            optimizer.zero_grad()
            if model_name[:4] == 'mtss':
                outputs = model(images, dems)
            else:
                images = torch.cat((images, dems), 1)
                outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            np_loss = loss.detach().cpu().numpy()
            interval_loss += np_loss
            if vis is not None:
                vis.vis_scalar('Loss', cur_itrs, np_loss)
             
            if (cur_itrs) % 10 == 0:
                interval_loss = interval_loss / 10
                print("Epoch %d, Itrs %d/%d, Loss=%.4f" %
                      (cur_epochs, cur_itrs, total_itrs, interval_loss))
                interval_loss = 0.0
                
            if (cur_itrs) % val_interval == 0:
                print("validation...")
                model.eval()
                val_score, ret_samples = validate(
                        model_name=model_name, model=model, loader=val_loader, device=device, metrics=metrics, ret_samples_ids=vis_sample_id)
                print(metrics.to_str(val_score))

                if val_score['Mean IoU'] > best_score:  # save best model
                    best_score = val_score['Mean IoU']
                    save_ckpt('checkpoints/mtss/best_%s-Score%.4f-Epoch%d-Iters%d-Total_Loss%.4f.pth' %
                          (model_name, best_score, cur_epochs, cur_itrs, interval_loss))
                # else:
                #     save_ckpt('checkpoints/villageLand/latest_%s-Score%.4f-Epoch%d-Iters%d-Total_Loss%.4f.pth' %
                #           (model_name, cur_score, cur_epochs, cur_itrs, interval_loss))

                if vis is not None:  # visualize validation score and samples
                    vis.vis_scalar("[Val] Overall Acc", cur_itrs, val_score['Overall Acc'])
                    vis.vis_scalar("[Val] Mean IoU", cur_itrs, val_score['Mean IoU'])
                    vis.vis_table("[Val] Class IoU", val_score['Class IoU'])
                if training_log == True:
                    file_handle = open('./train/logs/trainLog_villageLand.txt', mode='a+')
                    file_handle.writelines([
                    '*------------------------------------------------*\n'
                    'The prediction of model %s\n' %(model_name),
                    'Epoch: %d\n' %(cur_epochs),
                    'Itrs %d/%d\n' %(cur_itrs, total_itrs),
                    'Total_Loss_%f\n' %(interval_loss),
                    metrics.to_str(val_score)
                    ])
                    file_handle.close()
                model.train()
            scheduler.step()
            if cur_itrs >=  total_itrs:
                return


if __name__ == '__main__':
    train()
