import sys
sys.path.append(r'F:\code\python\SemanticSegmentation\autogis')
import cv2
import numpy as np
from PIL import Image
from pathlib import Path
from PyQt5 import QtGui, QtWidgets
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtWidgets import QFileDialog, QMessageBox
from PyQt5.QtCore import Qt, QBasicTimer


def mountain_extract(image, grad_we, grad_sn, division_coef=0.25, slope_threshold=9, mode='simple'):
    """
    提取图像中的山体，将高程数据分为山体和平原

    Parameters
    ----------
    image: ndarray
    grad_we: 格网宽度
    grad_sn: 格网高度
    slope_threshold: 坡度阈值，大于该阈值的会被初步划分为山地
    mode: {'simple', 'complex'}, optional, default='simple'
        simple: 不进行具体的山体类型划分，只划分山地和平原
        complex: 划分具体的山区类型

    Returns
    -------
    plain_hill: ndarray
        二值图像，山区为0，平原为1
    """
    if mode == 'simple':
        # 计算坡度
        slope = cal_slope(image, grad_we, grad_sn)
        m_maxdot = image.max()  # 区域最大高程
        m_eventdot = image.mean()   # 区域平均高程
        dividLine = m_eventdot +division_coef*(m_maxdot-m_eventdot)  # 高度分界线
        plain_hill = np.zeros_like(image, dtype=np.uint8)
        # 坡度小于9且高度低于分界线的为平原
        plain_hill[np.logical_and(slope<=slope_threshold, image<=dividLine)] = 1
    elif mode == 'complex':
        # 计算平面曲率和剖面曲率
        curv_kh, curv_kv = cal_curvature(image, method='dawei')
        # 计算坡度
        slope = cal_slope(image, grad_we, grad_sn)
        dtm_geo = np.zeros_like(image, dtype=np.uint8)
        # 地貌分类
        for i in range(image.shape[0]):
            for j in range(image.shape[1]):
                if(slope[i][j] < 2):
                    dtm_geo[i][j] = 1
                elif ((slope[i][j] >= 2) and (slope[i][j] <= 9)):
                    dtm_geo[i][j] = 2
                elif (curv_kv[i][j] <= 0 and curv_kh[i][j] >= 0 and slope[i][j] <= 15 and slope[i][j] > 9):
                    dtm_geo[i][j] = 3
                elif (curv_kv[i][j] <= 0 and curv_kh[i][j] >= 0 and slope[i][j] <= 45 and slope[i][j] > 15):
                    dtm_geo[i][j] = 4
                elif (curv_kv[i][j] > 0 and curv_kh[i][j] < 0 and slope[i][j] <= 45 and slope[i][j] >= 15):
                    dtm_geo[i][j] = 5
                elif (curv_kv[i][j] > 0 and curv_kh[i][j] > 0):
                    dtm_geo[i][j] = 6
        plain_hill = np.zeros_like(dtm_geo, dtype=np.uint8)
        m_maxdot = image.max()  # 区域最大高程
        m_eventdot = image.mean()   # 区域平均高程
        dividLine = m_eventdot +division_coef*(m_maxdot-m_eventdot)  # 高度分界线
        # 山区和平原分离，满足地势位于1或2级且高度小于分界线的为平原
        plain_location = np.logical_and(np.logical_or(dtm_geo==1, dtm_geo==2), image<=dividLine)
        plain_hill[plain_location] = 1
    return plain_hill

def mountain_adjust(image,  kernel_close=30, close_iter=2, kernel_open=30, open_iter=1,
                    plain_threshold=30000, hill_threshold=30000, mode='simple'):
    """
    调整计算得到的山区

    Parameters
    ----------
    image: ndarray
        初步提取出的山区二值图像
    kernel_close: int
        闭运算的卷积核大小，用于消除狭长的山地，值越大消除的山地越多
    close_iter: int
        闭运算的迭代次数
    kernel_open: int
        开运算的卷积核大小，用于消除狭长的平原，值越大消除的平原越多
    open_iter: int
        开运算的迭代次数
    plain_threshold: int
        山区调整时平原的面积阈值，小于该面积的会被删去，默认为30000*0.53*0.53 = 8427 (m2)
    hill_threshold: int
        山区调整时山区的面积阈值，小于该面积的会被删去，默认为30000*0.53*0.53 = 8427 (m2) 
    mode: {'simple', 'complex'}optional, default='simple'
        simple: 只进行开闭运算消融狭长的山地和狭长的平原
        complex： 消融较小的山地和较小的平原（使用四邻接种子填充法，极其消耗时间）

    Return
    ------
    image: ndarray
        调整后的结果
    """
    # 闭运算消融狭长的山地
    k1 = np.ones((kernel_close, kernel_close), np.uint8)
    move_longHill = cv2.morphologyEx(image, cv2.MORPH_CLOSE, k1, iterations=close_iter)

    # 将平原内面积较小的山地消融,种子填充
    move_smallHill = move_longHill.copy()
    if mode == 'complex':
        for i in range(move_longHill.shape[0]):
            for j in range(move_longHill.shape[1]):
                if move_longHill[i][j] == 0:
                    move_longHill, seed_r = regian_seedone4(move_longHill, i, j, 0, 1)
                    if seed_r[0][0] < hill_threshold:
                        move_smallHill = filling_gray(move_smallHill, seed_r, 1)

    # 开运算消除狭长的平原
    k2 = np.ones((kernel_open, kernel_open), np.uint8)
    move_longPlain = cv2.morphologyEx(move_smallHill, cv2.MORPH_OPEN, k2, iterations=open_iter)

    # 将山地内面积较小的平原消除
    move_smallPlain = move_longPlain.copy()
    if mode == 'complex':
        for i in range(move_longPlain.shape[0]):
            for j in range(move_longPlain.shape[1]):
                if move_longPlain[i][j] == 1:
                    move_longPlain, seed_r = regian_seedone4(move_longPlain, i, j, 1, 0)
                    if seed_r[0][0] < plain_threshold:
                        move_smallPlain = filling_gray(move_smallPlain, seed_r, 0)
    return move_smallPlain

def regian_seedone4(image, seed_x, seed_y, gray, aim_gray):
    """
    四邻接种子填充法

    Parameters
    ----------
    image: ndarray
    seed_x: int
        当前种子点的x坐标
    seed_y: int
        当前种子点的y坐标
    gray：uint8
        被填充点的像素值
    aim_gray: uint8
        填充的像素

    Return
    ------
    image: ndarray
        根据当前种子点，迭代填充后的图像
    seed_r: List
        填充过的像素点坐标，seed_r[0]为填充的像素点总数，
        seed_r[index]为填充点的坐标，类型为tuple，（seed_x, seed_y)
    """
    mark_number = 0     # 被填充点的个数
    iCurrentPixelx = 0
    iCurrentPixely = 0      # 当前像素位置
    seed_r = []     # 填充过的种子
    seed_r.append((mark_number, mark_number))

    seeds = []      # 种子堆栈
    # 初始化种子
    seeds.append((seed_x, seed_y))
    # 迭代查找可被填充的点
    while seeds:
        iCurrentPixelx = seeds[-1][0]
        iCurrentPixely = seeds[-1][1]
        seeds.pop(-1)
        image[iCurrentPixelx][iCurrentPixely] = aim_gray
        seed_r.append((iCurrentPixelx, iCurrentPixely))
        mark_number+=1
        # 判断左面的点，如果为gray，则压入堆栈
        if iCurrentPixely-1 >= 0:
            if image[iCurrentPixelx][iCurrentPixely-1] == gray:
                seeds.append((iCurrentPixelx, iCurrentPixely-1))
        # 判断下面的点，如果为gray，则压入堆栈
        if iCurrentPixelx+1 < image.shape[0]:
            if image[iCurrentPixelx+1][iCurrentPixely] == gray:
                seeds.append((iCurrentPixelx+1, iCurrentPixely))
        # 判断右面的点，如果为gray，则压入堆栈
        if iCurrentPixely+1 < image.shape[1]:
            if image[iCurrentPixelx][iCurrentPixely+1] == gray:
                seeds.append((iCurrentPixelx, iCurrentPixely+1))
        # 判断下面的点，如果为gray，则压入堆栈
        if iCurrentPixelx-1 >= 0:
            if image[iCurrentPixelx-1][iCurrentPixely] == gray:
                seeds.append((iCurrentPixelx-1, iCurrentPixely))
    seed_r[0] = (mark_number, mark_number)
    return image, seed_r

def filling_gray(image, seed_r, fill_gray):
    """
    将图像内指定区域的像素点进行填充

    Parameters
    ----------
    image: ndarray
    seed_r: List
        待填充区域，seed_r[0]为所要填充的像素点个数，
        seed_r[index]为该点的坐标
    fill_gray: uint8
        所要填充的颜色

    Return
    ------
    image_filled: ndarray
    """
    try:
        print(seed_r[0][0])
        for index in range(seed_r[0][0]):
            x = seed_r[index+1][0]
            y = seed_r[index+1][1]
            image[x][y] = fill_gray
    except Exception as e:
        print('error')
    return image

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

def adjust_mask(image):
    """
    调整掩膜用于提取山脚线

    Parameters
    ----------

    image: ndarray
        二值图像，平原为255， 山区为0
    
    Return
    ------
    result: ndarray
        二值图像，平原为0，山区为255，外边缘1层像素为0
    """
    mask = np.zeros_like(image, dtype=np.uint8)
    mask[image==0] = 255
    mask[0:10,:] = 0
    mask[-10:-1,:] = 0
    mask[:,0:10] = 0
    mask[:,-10:-1] = 0
    print(mask[0,:])
    return mask

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
    qim = QtGui.QImage(data, image.size[0], image.size[1], QtGui.QImage.Format_ARGB32)
    pixmap = QtGui.QPixmap.fromImage(qim)
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
    
def tif2bmp(img: Image)->Image:
    """
    将高程tif文件转为bmp文件，压缩方法：
        255 * (currentNum - minNum)/(maxNum - minNum)
    """
    dem_array = np.array(img)
    dem_min = dem_array.min()
    dem_max = dem_array.max()
    dem_array = 255 * np.divide(dem_array - dem_min, dem_max - dem_min)
    dem_array = dem_array.astype(np.uint8)
    result = Image.fromarray(dem_array)
    return result

def simple_extract(img_path, grad_we=0.53,grad_sn=0.53, show=True)->Image:
    """
    提取单张图片

    Parameters
    ----------
    img_path: Path
        图片路径
    show:{True, False}optionls
        是否显示图片
    
    Return
    ------
    image: Image
        PIL图片对象
    """
    image = Image.open(img_path)
    image = np.array(image)
    image = mountain_extract(image, grad_we, grad_sn)
    image = mountain_adjust(image, mode='simple')
    # image = mountain_adjust(image, mode='complex')
    image[image==1] = 255
    result = Image.fromarray(image)
    if show == True:
        result.show()
    return result

def iter_extract(forder_path, target_path):
    """
    提取文件夹内的所有图片

    Parameters
    ----------
    forder_path: Path
        待提取的文件夹
    target_path: Path
        目标文件夹，用于保存提取结果
    """
    for _, filePath in enumerate(forder_path.iterdir()):
        fileName = filePath.stem
        print(fileName)
        result = simple_extract(filePath, show=False)
        target_path = forder_path / (fileName + '.png')
        result.save(target_path, quality=95)

class BaseMainWindow(QtWidgets.QMainWindow):
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

if __name__ == '__main__':  
    # file_path = Path('interface/mountain_foot/data/bagui_dem.tif')
    folder_path = Path(r'F:\dataset\villageLand_original\DEMImages_tif')
    target_path = Path(r'F:\dataset\villageLand_original\mountains')
    # simple_extract(file_path)
    # iter_extract(folder_path, target_path)

