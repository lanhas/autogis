# coding: utf-8
import os
from PIL import Image
import torch
import numpy as np
import cv2
import matplotlib.pyplot as plt


def draw_CAM(model, img_path, save_path, decode_fn=None, visual_heatmap=False):
    '''
    绘制 Class Activation Map
    :param model: 加载好权重的Pytorch model
    :param img_path: 测试图片路径
    :param save_path: CAM结果保存路径
    :param transform: 输入图像预处理方法
    :param visual_heatmap: 是否可视化原始heatmap（调用matplotlib）
    :return:
    '''
    # 图像加载&预处理
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    ori_image = np.array(Image.open(img_path).resize((128, 128)))
    image = np.expand_dims(ori_image, axis=(0, 1))
    image = torch.Tensor(image / 255.0).to(device)
    # img = Image.open(img_path).convert('RGB')
    # if transform:
    #     img = transform(img)
    # img = img.unsqueeze(0)

    # 获取模型输出的feature/score
    model.eval()
    features = model.encoder(image)
    output = model.classifier(features.view(features.shape[0], features.shape[1], -1).mean(dim=2))

    # 为了能读取到中间梯度定义的辅助函数
    def extract(g):
        global features_grad
        features_grad = g

    # 预测得分最高的那一类对应的输出score
    pred = torch.argmax(output).item()
    pred_class = output[:, pred]

    features.register_hook(extract)
    pred_class.backward()  # 计算梯度

    grads = features_grad  # 获取梯度

    pooled_grads = torch.nn.functional.adaptive_avg_pool2d(grads, (1, 1))

    # 此处batch size默认为1，所以去掉了第0维（batch size维）
    pooled_grads = pooled_grads[0]
    features = features[0]
    # 512是最后一层feature的通道数
    for i in range(512):
        features[i, ...] *= pooled_grads[i, ...]

    # 以下部分同Keras版实现
    heatmap = features.cpu().detach().numpy()
    heatmap = np.mean(heatmap, axis=0)

    heatmap = np.maximum(heatmap, 0)
    heatmap /= np.max(heatmap)

    # 可视化原始热力图
    if visual_heatmap:
        plt.matshow(heatmap)
        plt.show()

    if decode_fn is not None:
        image = decode_fn[ori_image.astype('uint8')]
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    # img = cv2.imread(img_path)  # 用cv2加载原始图像
    heatmap = cv2.resize(heatmap, (image.shape[1], image.shape[0]))  # 将热力图的大小调整为与原始图像相同
    heatmap = np.uint8(255 * heatmap)  # 将热力图转换为RGB格式
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)  # 将热力图应用于原始图像
    superimposed_img = heatmap * 0.4 + image  # 这里的0.4是热力图强度因子
    cv2.imwrite(str(save_path), superimposed_img)  # 将图像保存到硬盘


def draw_CAM_mm(model, img_path, dem_path, save_path, decode_fn=None, visual_heatmap=False):
    # 图像加载&预处理
    # temp
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # ori_remote = np.array(Image.fromarray(decode_fn[np.array(Image.open(img_path)).astype('uint8')]).resize((128, 128)))
    ori_remote = np.array(Image.open(img_path).resize((128, 128)))
    # decode_fn[np.array(Image.open(x)).astype('uint8')]

    remote = ori_remote.transpose(2, 0, 1)
    dem = np.expand_dims(np.array(Image.open(dem_path).resize((128, 128))), axis=2).transpose(2, 0, 1)
    image = np.expand_dims(np.concatenate((remote, dem), axis=0), axis=0)
    # image = np.expand_dims(remote, axis=0)
    image = torch.Tensor(image / 255.0).to(device)

    # 获取模型输出的feature/score
    model.eval()
    features = model.encoder(image)
    output = model.classifier(features.view(features.shape[0], features.shape[1], -1).mean(dim=2))

    # 为了能读取到中间梯度定义的辅助函数
    def extract(g):
        global features_grad
        features_grad = g

    # 预测得分最高的那一类对应的输出score
    pred = torch.argmax(output).item()
    pred_class = output[:, pred]

    features.register_hook(extract)
    pred_class.backward()  # 计算梯度

    grads = features_grad  # 获取梯度

    pooled_grads = torch.nn.functional.adaptive_avg_pool2d(grads, (1, 1))

    # 此处batch size默认为1，所以去掉了第0维（batch size维）
    pooled_grads = pooled_grads[0]
    features = features[0]
    # 512是最后一层feature的通道数
    for i in range(512):
        features[i, ...] *= pooled_grads[i, ...]

    # 以下部分同Keras版实现
    heatmap = features.cpu().detach().numpy()
    heatmap = np.mean(heatmap, axis=0)

    heatmap = np.maximum(heatmap, 0)
    heatmap /= np.max(heatmap)

    # 可视化原始热力图
    if visual_heatmap:
        plt.matshow(heatmap)
        plt.show()

    # if decode_fn is not None:
        # image = decode_fn[ori_remote.astype('uint8')]
    image = cv2.cvtColor(ori_remote, cv2.COLOR_RGB2BGR)

    # img = cv2.imread(img_path)  # 用cv2加载原始图像
    heatmap = cv2.resize(heatmap, (image.shape[1], image.shape[0]))  # 将热力图的大小调整为与原始图像相同
    heatmap = np.uint8(255 * heatmap)  # 将热力图转换为RGB格式
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)  # 将热力图应用于原始图像
    superimposed_img = heatmap * 0.4 + image  # 这里的0.4是热力图强度因子
    cv2.imwrite(str(save_path), superimposed_img)  # 将图像保存到硬盘