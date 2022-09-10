import sys
sys.path.append(r'F:\code\python\SemanticSegmentation\autogis')
import os
import cv2
import torch
import models
import numpy as np
import torch.nn as nn
import matplotlib.pyplot as plt
from pathlib import Path
from PIL import Image
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtWidgets import QFileDialog, QMessageBox, QMainWindow
from datasets.village_segm import villageFactorsSegm
import torchvision.transforms as T

class MtssPredicter():
    """
    多模态地理要素语义分割，用于获取村落地理要素特征图
    """
    def __init__(self, pmode, backbone, device, pmethod, resize, num_class) -> None:
        self.decode_fn = villageFactorsSegm.decode_target
        if device == 'cuda':
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device('cpu')
        if pmode == 'M':
            self.model_name = 'mtss_' + backbone
        else:
            self.model_name = 'deeplabv3plus_' + backbone
        self.predictMode = 'singleImg'                  # 默认预测模式
        self.resize = resize
        self.num_classes = num_class                    # 土地类别
        self.embeddings = None                          # 中心点的坐标
        self.output_stride = 16                         # deeplab的步幅，用于控制ASPP中的卷积步幅
        self.model_map = {
        'mtss_resnet50': models.mtss_resnet50,
        'mtss_resnet101': models.mtss_resnet101,
        'mtss_mobilenet': models.mtss_mobilenet,
        'deeplabv3plus_resnet50': models.deeplabv3plus_resnet50,
        'deeplabv3plus_resnet101': models.deeplabv3plus_resnet101,
        'deeplabv3plus_mobilenet': models.deeplabv3plus_mobilenet
        }
        self.transform_remote = T.Compose([
                T.ToTensor(),
                T.Normalize(mean=[0.2737, 0.3910, 0.3276],
                            std=[0.1801, 0.1560, 0.1301]),
            ])
        self.transform_dem = T.Compose([
                T.ToTensor(),
                T.Normalize(mean=[0.4153],
                            std=[0.2405]),
            ])

    def load_model(self, ckpt):
        # try:
        if ckpt is not None and os.path.isfile(ckpt):
            self.model = self.model_map[self.model_name](num_classes=self.num_classes, output_stride=self.output_stride)
            checkpoint = torch.load(ckpt, map_location=torch.device('cpu'))
            self.model.load_state_dict(checkpoint["model_state"], False)
            model = nn.DataParallel(self.model)
            model.to(self.device)
            del checkpoint
            return True
        else:
            return False
        # except Exception as e:
        #     QMessageBox.warning(self, "警告", "预测失败，请重试！\n 错误：{}".format(e), QMessageBox.Ok)
        #     print(e)
    
    def predict(self, image, dem):
        """
        预测函数，将遥感图像送入网络进行土地分类

        Parameters
        ----------
        image: Image
            输入的遥感图像，PIL的Image类型

        Return
        ------
        result: Image
            预测结果，单通道预测结果
        """
        try:
            with torch.no_grad():
                self.model = self.model.eval()
                image = image.convert('RGB')
                image = self.transform_remote(image).unsqueeze(0) # To tensor of NCHW
                image = image.to(self.device)

                dem = self.transform_dem(dem).unsqueeze(0) # To tensor of NCHW
                dem = dem.to(self.device)

                result = Image.fromarray(self.model(image, dem).max(1)[1].cpu().numpy()[0]) # HW
            return result
        except Exception as e:
            QMessageBox.warning(self, "警告", "预测失败，请重试！\n 错误：{}".format(e), QMessageBox.Ok)
            print(e)

class MtvcPredicter():
    """
    村落自动分类，用于根据村落地理要素特征图得到村落类别
    """
    def __init__(self, device, size, num_class) -> None:
        if device == 'cuda':
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device('cpu')
        self.model_name = 'siameseNetwork'
        self.size = (512, 512)
        self.num_class = num_class
        self.model_map = {
        'classificationNet': models.classificationNet,
        'siameseNetwork': models.siameseNetwork,
        'tripletNetwork': models.tripletNetwork,
        'onlinePairSelection': models.onlinePairSelection,
        'onlineTripletSelection': models.onlineTripletSelection,
    }
        self.transform = T.Compose([
                T.Resize(size=self.size),
                T.ToTensor(),
                T.Normalize(mean=[0.485],
                        std=[0.229]),
                ])

    def load_model(self, ckpt):
        # try:
        if ckpt is not None and os.path.isfile(ckpt):
            self.model = self.model_map[self.model_name]()
            checkpoint = torch.load(ckpt, map_location=torch.device('cuda'))
            self.model.load_state_dict(checkpoint["model_state"])
            model = nn.DataParallel(self.model)
            model.to(self.device)
            del checkpoint
            return True
        else:
            return False
        # except Exception as e:
        #     QMessageBox.warning(self, "警告", "预测失败，请重试！\n 错误：{}".format(e), QMessageBox.Ok)
        #     print(e)

    def predict(self, image):
        center_embeddings = self.get_centrePoint()
        with torch.no_grad():
            image = self.transform(image).unsqueeze(0)
            image = image.to(self.device)
            self.model = self.model.eval()
            embedding = self.model.get_embedding(image).data.cpu().numpy()[0]
            dists = [np.sqrt(np.sum(np.square(embedding - center_embeddings[i]))) for i in range(5)]
            label = dists.index(min(dists)) + 1
            # 绘制中心点
            plt.figure(figsize=(10, 10))
            for i in range(self.num_class):
                plt.scatter(center_embeddings[i, 0], center_embeddings[i, 1], alpha=0.5, s=200, color=village_colors[i+1], marker="*")
            # 绘制当前点
            plt.scatter(embedding[0], embedding[1], alpha=0.5, color=village_colors[label])
            plt.legend(village_classes)
            plt.show()

    def test(self):
        from predict.predict_mtvc import test
        test()

    def get_centrePoint(self):
        paths = [Path.cwd() / 'datasets/mtvcd/villageMask' / path for path in normal_village]
        images = [Image.open(path) for path in paths]
        with torch.no_grad():
            images = list(map(lambda x: self.transform(x).unsqueeze(0), [x for x in images]))
            image = torch.from_numpy(np.concatenate(images, axis=0))
            image = image.to(self.device)
            self.model = self.model.eval()
            embeddings = self.model.get_embedding(image).data.cpu().numpy()
        return embeddings

    def extract_embeddings(self, dataloader, model):
        with torch.no_grad():
            model.eval()
            embeddings = np.zeros((len(dataloader.dataset.labels), 2))
            labels = np.zeros(len(dataloader.dataset.labels))
            k = 0
            for images, target in dataloader:
                if torch.cuda.is_available():
                    images = torch.from_numpy(np.array(images))
                embeddings[k:k+len(images)] = model.get_embedding(images).data.cpu().numpy()
                labels[k:k+len(images)] = np.array(target)
                k += len(images)
        return embeddings, labels

    def plot_embeddings(self, embeddings, targets, num_classes, xlim=None, ylim=None):
        plt.figure(figsize=(10, 10))
        for i in range(1, num_classes+1):
            inds = np.where(targets==i)[0]
            plt.scatter(embeddings[inds, 0], embeddings[inds,1], alpha=0.5, color=village_colors[i])
        if xlim:
            plt.xlim(xlim[0], xlim[1])
        if ylim:
            plt.ylim(ylim[0], ylim[1])
        plt.legend(village_classes)
        plt.show()
    

class BaseMainWindow(QMainWindow):
    """对QDialog类重写，实现一些功能"""

    def closeEvent(self, event):
        """
        重写closeEvent方法，实现dialog窗体关闭时执行一些代码
        :param event: close()触发的事件
        :return: None
        """
        reply = QMessageBox.question(self, '本程序',
                                        "是否要退出界面？",
                                        QMessageBox.Yes | QMessageBox.No,
                                        QMessageBox.No)
        if reply == QMessageBox.Yes:
            event.accept()
        else:
            event.ignore()

def image_blend(image, areaMask, alpha, beta, gamma) -> Image:
    """
    将图片内的前背景按一定比例区分

    Parameters
    ----------
    image: ndarray
    areaMask: ndarray
        区域掩膜
    alpha: float
        前景区的融合比例
    beta: float
        背景区的融合比例
    gamma: 透明度

    Return
    ------
    result: ndarray
        融合后的结果
    """
    foreground = image
    background = image.copy()
    # 如果掩膜是单通道图像，先将其转为三通道
    if len(areaMask.shape) == 2:
        for i in range(3):
            foreground[:, :, i][areaMask == 0] = 0
            background[:, :, i][areaMask > 0] = 0
    result = cv2.addWeighted(foreground, alpha, background, beta, gamma)
    return result

def color2annotation(image) -> np.array:
    """
    将三通道的颜色label转为单通道的annotation

    Parameters
    ----------
    image: ndarray

    Return
    ------
    annotation: ndarray
    """
    image = np.array(decode_fn(image), np.uint8)
    annotation = np.zeros_like(image[:, :, 0], dtype=np.uint8)
    annotation[cv2.inRange(image, colorMask['village'], colorMask['village'])==255] = 1
    annotation[cv2.inRange(image, colorMask['mountain'], colorMask['mountain'])==255] = 2
    annotation[cv2.inRange(image, colorMask['water'], colorMask['water'])==255] = 3
    annotation[cv2.inRange(image, colorMask['forest'], colorMask['forest'])==255] = 4
    annotation[cv2.inRange(image, colorMask['farmland'], colorMask['farmland'])==255] = 5
    annotation[cv2.inRange(image, colorMask['wasteland'], colorMask['wasteland'])==255] = 6
    annotation[cv2.inRange(image, colorMask['unknown'], colorMask['unknown'])==255] = 7
    return annotation

def getAreaMask(image, areaIndex):
    """
    根据类别标签，获得annotation对应的二值掩膜，0代表背景，1代表该区域

    Parameters
    ----------
    image: ndarray
        图像的label，三通道彩色图像
    areaIndex: int
        区域标签1~类别总数
        village             1
        mountain            2
        water               3
        forest              4
        farmland            5
        wasteland           6
        unknown             7
    Return
    ------
    result: ndarray
        二值掩膜，单通道，0代表背景，1代表该区域
    """
    # 三通道彩色label转单通道annotation
    result = np.zeros_like(image, dtype=np.uint8)
    if areaIndex == 0:
        result = image
    elif areaIndex == 1:
        result[image == 1] = 1
    elif areaIndex == 2:
        result[image == 2] = 1
    elif areaIndex == 3:
        result[image == 3] = 1
    elif areaIndex == 4:
        result[image == 4] = 1
    elif areaIndex == 5:
        result[image == 5] = 1
    elif areaIndex == 6:
        result[image == 6] = 1
    elif areaIndex == 7:
        result[image == 7] = 1
    return result

def img_addition(image, areaMask, axisColor):
    """
    为图片内掩膜区域上色

    Parameters
    ----------
    image: ndarray
    areaMask: ndarray
        区域掩膜
    axisColor: tuple
        颜色，(r, g, b)

    Return
    ------
    image: ndarray
    """
    image[:, :, 0][areaMask > 0] = axisColor[0]
    image[:, :, 1][areaMask > 0] = axisColor[1]
    image[:, :, 2][areaMask > 0] = axisColor[2]
    return image

def pil2pixmap(image):
    """
    将PIL Image类型转为Qt QPixmap类型
    """
    if image.mode == "RGB":
        r, g, b = image.split()
        image = Image.merge("RGB", (b, g, r))
    elif  image.mode == "RGBA":
        r, g, b, a = image.split()
        image = Image.merge("RGBA", (b, g, r, a))
    elif image.mode == "L":
        image = image.convert("RGBA")
    # Bild in RGBA konvertieren, falls nicht bereits passiert
    im2 = image.convert("RGBA")
    data = im2.tobytes("raw", "RGBA")
    qim = QImage(data, image.size[0], image.size[1], QImage.Format_ARGB32)
    pixmap = QPixmap.fromImage(qim)
    return pixmap

def ndarray2pixmap(ndarray):
    """
    将ndarray类型转为QPixmap类型
    """
    if len(ndarray.shape) == 3:
        height, width, channels = ndarray.shape
        bytesPerLine = 3 * width
        qImg = QImage(ndarray.data, width, height, bytesPerLine, QImage.Format_RGB888)
        qpix = QPixmap(qImg)
    else:
        height, width = ndarray.shape
        bytesPerLine = 3 * width
        qImg = QImage(ndarray.data, width, height, bytesPerLine, QImage.Format_RGB32)
        qpix = QPixmap(qImg)
    return qpix

def decode_fn(image):
    image = np.array(image)
    cmap = village_cmap()
    result = Image.fromarray(cmap[image].astype('uint8'))
    return result

colorMask = {'village': (0, 128, 128),
             'mountain': (128, 0, 0),
             'water': (0, 0, 128),
             'forest': (0, 128, 0),
             'farmland': (128, 128, 0),
             'wasteland': (128, 0, 128),
             'unknown': (0, 0, 0),
            }
village_classes = ['mountain ring of water around', 'adjoin mountain', 'along river', 'plain', 'mountain'] 
                # ['山环水绕型', '依山型', '沿河型', '平原型', '山地型']
normal_village = ['dadong.png', 'lingshan.png', 'tangzhai.png', 'yongxing.png', 'baiqiao.png']

village_colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b']

def village_cmap(N=7, normalized=False):
    dtype = 'float32' if normalized else 'uint8'
    cmap = np.zeros((N, 3), dtype=dtype)
    cmap[0] = [0, 0, 0]         # 未知土地：黑色，云层遮盖或其他因素
    cmap[1] = [128, 0, 0]       # 山体：地形上突出的部分，包括荒山、林山、农山等
    cmap[2] = [0, 128, 0]       # 森林/植被： 绿色，平原地区且被绿色包裹的非农田区域
    cmap[3] = [128, 128, 0]     # 农田：黄色，平原地区的农业区域
    cmap[4] = [0, 0, 128]       # 水系：深蓝色，江河湖海湿地
    cmap[5] = [128, 0, 128]     # 荒地：紫色
    cmap[6] = [0, 128, 128]     # 村落：浅蓝色，村庄
    cmap = cmap / 255 if normalized else cmap
    return cmap