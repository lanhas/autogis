import cv2
import numpy as np
from enum import Enum
from PIL import Image, ImageOps
from pathlib import Path

from PyQt5.QtWidgets import QFileDialog, QMainWindow, QApplication, QMessageBox
from PyQt5.QtCore import Qt, QPoint, QEvent
from PyQt5.QtGui import QImage, QPixmap, QPainter, QPen, QColor
from PIL.ImageQt import ImageQt


class EventType(Enum):
    """
    事件类型
    """
    noneType = 1        # 禁用鼠标事件
    drawOutline = 2     # 手动绘制边界线
    loadOutline = 3     # 加载边界线
    drawRoad = 4        # 道路绘制
    extractColor = 5    # 提取颜色


class OutlineColor(Enum):
    """
    边界线颜色，用于提取轮廓线
    """
    red = 1
    orange = 2
    yellow = 3
    green = 4
    cyan = 5
    blue = 6
    purple = 7
    black = 8
    gray = 9
    white = 10


Zhcn2ColorDict = {
    '红色': 'red',
    '橙色': 'orange',
    '黄色': 'yellow',
    '绿色': 'green',
    '青色': 'cyan',
    '蓝色': 'blue',
    '紫色': 'purple',
    '黑色': 'black',
    '白色': 'gray',
    '灰色': 'white',
}

# 颜色范围字典，用于确定取色结果属于哪种颜色((颜色下界), (颜色上界))、
colorScopeDict = {'red_1': ((0, 43, 46), (10, 255, 255)), 
                  'red_2': ((156, 43, 46), (180, 255, 255)),
                  'orange': ((11, 43, 46), (25, 255, 255)),
                  'yellow': ((26, 43, 46), (34, 255, 255)),
                  'green': ((35, 43, 46), (77, 255, 255)),
                  'cyan': ((78, 43, 46), (99, 255, 255)),
                  'blue': ((100, 43, 46), (124, 255, 255)),
                  'purple': ((125, 43, 46), (155, 255, 255)),
                  'black': ((0, 0, 0), (180, 255, 46)),
                  'gray': ((0, 0, 46), (180, 43, 220)),
                  'white': ((0, 0, 221), (180, 30, 255)),
            }

# 颜色字典，用于绘制线条，轴线显示等
colorDict = {'红色': (255, 0, 0),
             '橙色': (255, 165, 0),
             '黄色': (255, 255, 0),
             '绿色': (127, 255, 0),
             '青色': (0, 255, 255),
             '蓝色': (30, 144, 255),
             '紫色': (160, 32, 240),
             '黑色': (0, 0, 0),
             '灰色': (192, 192, 192),
             '白色': (255, 255, 255),
}


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


def getOutlineMask(image, outlineColor):
    """
    根据轮廓线的颜色，提取出轮廓

    Parameters
    ----------
    image: ndarray
        带有边界线的图像
    outlineColor: enum
        轮廓线线的颜色

    Return
    ------
    result: ndarray
        轮廓线掩膜，单通道图像，线条位置为1，背景为0
        :param image:
        :param outlineColor:
        :return:
    """
    # 转为hsv图像
    im_hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
    # 结果初始化
    result = np.zeros_like(image[:, :, 0], dtype=np.uint8)
    if outlineColor == OutlineColor.red:
        # cv2.inRange函数，根据颜色范围,得到该范围内颜色的位置
        red_location1 = cv2.inRange(im_hsv, colorScopeDict['red_1'][0], colorScopeDict['red_1'][1])==255
        red_location2 = cv2.inRange(im_hsv, colorScopeDict['red_2'][0], colorScopeDict['red_2'][1])==255
        result[np.logical_or(red_location1, red_location2)] = 1
    elif outlineColor == OutlineColor.orange:
        result[cv2.inRange(im_hsv, colorScopeDict['orange'][0], colorScopeDict['orange'][1])==255] = 1
    elif outlineColor == OutlineColor.yellow:
        result[cv2.inRange(im_hsv, colorScopeDict['yellow'][0], colorScopeDict['yellow'][1])==255] = 1
    elif outlineColor == OutlineColor.green:
        result[cv2.inRange(im_hsv, colorScopeDict['green'][0], colorScopeDict['green'][1])==255] = 1
    elif outlineColor == OutlineColor.cyan:
        result[cv2.inRange(im_hsv, colorScopeDict['cyan'][0], colorScopeDict['cyan'][1])==255] = 1
    elif outlineColor == OutlineColor.blue:
        result[cv2.inRange(im_hsv, colorScopeDict['blue'][0], colorScopeDict['blue'][1])==255] = 1
    elif outlineColor == OutlineColor.purple:
        result[cv2.inRange(im_hsv, colorScopeDict['purple'][0], colorScopeDict['purple'][1])==255] = 1
    elif outlineColor == OutlineColor.black:
        result[cv2.inRange(im_hsv, colorScopeDict['black'][0], colorScopeDict['black'][1])==255] = 1
    elif outlineColor == OutlineColor.gray:
        result[cv2.inRange(im_hsv, colorScopeDict['gray'][0], colorScopeDict['gray'][1])==255] = 1
    elif outlineColor == OutlineColor.white:
        result[cv2.inRange(im_hsv, colorScopeDict['white'][0], colorScopeDict['white'][1])==255] = 1
    return result

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

def pixmap2array(pixmap):
    ## Get the size of the current pixmap
    size = pixmap.size()
    h = size.width()
    w = size.height()

    ## Get the QImage Item and convert it to a byte string
    qimg = pixmap.toImage()
    byte_str = qimg.bits().tobytes()

    ## Using the np.frombuffer interface to convert the byte string into an np array
    img = np.frombuffer(byte_str, dtype=np.uint8).reshape((w,h,4))
    return img

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

def dilate_iter(image,villageMask, iter_num: int, kernelSize, line_width):
    image = image.astype('uint8')
    img_scope = np.array(villageMask)
    temp_img = image
    kernel = np.ones((kernelSize, kernelSize), np.uint8)
    kernel2 = np.ones((line_width, line_width), np.uint8)
    img = cv2.dilate(temp_img, kernel2, iterations=1)
    imgs = [img]
    for i in range(iter_num):
        img = cv2.dilate(temp_img, kernel, iterations=1)
        img[img_scope == 0] = 0
        imgs.append(img)
        temp_img = img
    imgs.append(img_scope)
    imgs.reverse()
    return imgs

def cal_slope(image, grad_we, grad_sn):
    """
    计算一张图片的坡度，使用三阶不带权差分法计算坡度

    Parameters
    ----------
    img_array: ndarray
    grad_we: float
        dem格网宽度，每像素代表的距离（单位：米)
    grad_sn: float
        dem格网高度，每像素代表的距离（单位：米）
    
    Return
    ------
    slope: ndarray
        坡度图
    """
    kernal_we = np.array([[1, 0, -1],
                          [2, 0, -2],
                          [1, 0, -1]])
    kernal_sn = np.array([[-1,-2,-1],
                          [0, 0, 0],
                          [1, 2, 1]])
    img = AddRound(image)
    slope_we = cv2.filter2D(img, -1, kernal_we)
    slope_sn = cv2.filter2D(img, -1, kernal_sn)
    slope_we=slope_we[1:-1,1:-1] / 8 / grad_we
    slope_sn=slope_sn[1:-1,1:-1] / 8 / grad_sn
    slope = (np.arctan(np.sqrt(slope_we*slope_we+slope_sn*slope_sn)))*57.29578
    return slope

def AddRound(image):
    """
    在图像的周围填充像素，填充值与边缘像素相同

    Parameters
    ----------
    image: ndarray

    Return
    ------
    addrounded_image: ndarray
    """
    ny, nx = image.shape  # ny:行数，nx:列数
    result=np.zeros((ny+2,nx+2))
    result[1:-1,1:-1]=image
    #四边
    result[0,1:-1]=image[0,:]
    result[-1,1:-1]=image[-1,:]
    result[1:-1,0]=image[:,0]
    result[1:-1,-1]=image[:,-1]
    #角点
    result[0,0]=image[0,0]
    result[0,-1]=image[0,-1]
    result[-1,0]=image[-1,0]
    result[-1,-1]=image[-1,0]
    return result

def cal_curvature(image, method='conv'):
    """
    计算一张图像的曲率

    Parameters
    ----------
    img_array : ndarray
    method : {'conv', 'derivation', 'dawei}, optional
        conv使用一次卷积操作，求得平均曲率
        derivation使用二阶偏导，求得平均曲率
        dawei使用二阶偏导，求得平面曲率curv_kh和剖面曲率curv_kv

    Returns
    -------
    'conv'与'derivation'返回平均曲率，类型ndarray
    'dawei'返回平面曲率curv_kh和剖面曲率curv_kv，类型tuple

    """
    if method == 'conv':
        kernal = np.array([[-1/16, 5/16, -1/16],
                            [5/16, -1, 5/16],
                            [-1/16, 5/16, -1/16]])
        final = cv2.filter2D(image, -1, kernal) 
    elif method == 'derivation':
        x , y = np.gradient(image)
        xx, xy = np.gradient(x)
        yx, yy = np.gradient(y)
        Iup =  (1+x*x)*yy - 2*x*y*xy + (1+y*y)*xx
        Idown = np.power((2*(1 + x*x + y*y)),1.5) 
        final = Iup/Idown
        final=abs(final)
        final = (final-final.min())/(final.max()-final.min())
        final = final * 255
        final = final.astype(np.uint8)
    elif method == 'dawei':
        curv_kh = np.zeros_like(image)
        curv_kv = np.zeros_like(image)
        x , y = np.gradient(image)
        xx, xy = np.gradient(x)
        yx, yy = np.gradient(y)
        Idown = x*x + y*y*np.sqrt(1+x*x+y*y)
        # if not np.any(Idown==0):
        kh_Iup = -(y*y*xx-2*x*y*xy+x*x*yy)
        kv_Iup = -(x*x*xx+2*x*y*xy+y*y*yy)
        curv_kh = kh_Iup / Idown
        curv_kv = kv_Iup / Idown
        curv_kh[np.isnan(curv_kh)] = 0
        curv_kv[np.isnan(curv_kv)] = 0
        final = (curv_kh, curv_kv)
    return final

def tif2bmp(image):
    """
    将tif转为rgb位图
    """
    image = np.array(image)
    _max = np.max(image)
    _min = np.min(image)
    image = ((image - _min) / (_max - _min)) * 255
    image = Image.fromarray(image)
    image = image.convert('L')
    return image

if __name__ == '__main__':
    import matplotlib.pyplot as plt
    dem_dbs = Image.open(r'utils\imgaug\data\shan.jpg')
    box = (0, 0, 400, dem_dbs.size[1])
    result = dem_dbs.crop(box)
    print(result.size)
    result.show()

