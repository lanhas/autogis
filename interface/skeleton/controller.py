import time
import io
from PyQt5.QtWidgets import QWidget
from PyQt5.QtCore import pyqtSignal
from interface.skeleton.skeleton import Ui_MainWindow as skeletonWindow
from skimage.morphology import medial_axis, skeletonize
from interface.skeleton.parameters import Ui_Form as paraWindow
from interface.skeleton.func import *


class Skeleton(BaseMainWindow, skeletonWindow):
    """
    村落骨架提取，包括中轴变换与图像细化技术
    """
    def __init__(self, parent=None) -> None:
        super(Skeleton, self).__init__(parent=parent)
        self.setupUi(self)
        self.eventType = EventType.noneType     # 事件类型，用于控制鼠标事件
        # 初始化
        self.originalImg = None     # 原始图像
        self.villageMask = None     # 村落掩膜，通过村落边界或山水轮廓线获得的村落区域
        self.pixmap = None          # 显示在qlabel中的pixmap图
        self.slopeImg = None        # 坡度图
        self.curvatureImg = None    # 曲率图
        self.slopeDivided = None    # 坡度小于坡度阈值的区域
        self.elevationData = None   # 村落高程数据，用于辅助中轴线生成
        self.outlineImg = None      # 带有村落边界线的图像，用于确定村落区域
        self.resultImg = None       # 结果图像，融合骨架线和原始图像后的结果
        self.outlinePix = None      # 绘制村落边界线的中间过程图像
        self.skPix = None           # 骨架线的中间过程图像
        self.img_name = None        # 图片名
        self.showing_pixmap = None     # 正在label中显示的图片
        # 默认参数
        self.gradWe = 8.5     # dem格网宽度，0.53米/像素
        self.gradSn = 8.5     # dem格网高度，0.53米/像素
        self.slope_threshold = 15       # 坡度阈值
        self.kernelSize = 15     # 核大小 
        self.iterNum = 10       # 迭代次数
        self.sleepTime = 0.0001      # 控制动态显示的间隔时间
        # 画笔颜色
        self.contourPenCol = QColor('#FF0000')      # 轮廓线，默认为红色
        self.roadPenCol = QColor('#33FFFF')         # 道路线，默认为蓝色
        # 轴线三通道值
        self.axisColor = colorDict['橙色']       # 轴线颜色，默认为橙色
        self.axisWidth = 7                          # 轴线宽度
        # 边界线颜色
        self.outlineColor = OutlineColor.red        # 边界线提取时的默认颜色，默认为红色
        self.paraWindow = ParaWindow(self.gradSn, self.gradWe, self.kernelSize, self.iterNum, 
                self.slope_threshold, self.sleepTime, self.contourPenCol, self.roadPenCol, self.axisColor, self.axisWidth, self.outlineColor)
        self.paraWindow.para_commit.connect(self.update_parameters)

    def open_file(self):
        """
        读取遥感图像
        """
        self.eventType = EventType.noneType
        try:
            fname ,_ = QFileDialog.getOpenFileName(self,'Open File', str(Path.home() / "Pictures"),
                                                    'Image files (*.jpg *.tif *.tiff *.png *.jpeg)')
            if fname != '':
                image = Image.open(fname)
                self.img_name = Path(fname).stem
                # 将图片按照原始比例显示在label
                self.label_show(image)
                self.label.setMinimumSize(1, 1)
                self.label.installEventFilter(self)
                self.originalImg = image
                # temp code start
                self.eventType = EventType.loadOutline
                self.outlineImg = image
                # temp code end
                self.empty_result()
        except Exception as e:
            QMessageBox.warning(self, '提示', '打开图片失败，请检查图片类型和图片大小！', QMessageBox.Ok)
            print(e)

    def openOutline(self):
        """
        读取边界线图像
        """
        self.eventType = EventType.loadOutline
        try:
            if self.originalImg is None:
                QMessageBox.warning(self, '提示', '请先添加原图！', QMessageBox.Ok)
            else:
                fname ,_ = QFileDialog.getOpenFileName(self,'Open File', str(Path.home() / "Pictures"),
                                                        'Image files (*.jpg *.tif *.tiff *.png *.jpeg)')
                if fname != '':
                    image = Image.open(fname)
                    self.label_show(image)
                    self.outlineImg = image
                    self.empty_result()
        except Exception as e:
            QMessageBox.warning(self, '提示', '打开轮廓线失败，请检查图片类型和图片大小！', QMessageBox.Ok)

    def add_elevationData(self):
        """
        添加高程数据
        """
        self.eventType = EventType.noneType
        try:
            fname ,_ = QFileDialog.getOpenFileName(self,'Open elevation File', str(Path.home() / "Pictures"),
                                                    'Image files (*.jpg *.tif *.tiff *.png *.jpeg)')
            if fname != '':
                image = Image.open(fname)
                image = tif2bmp(image)
                self.label_show(image)
                self.elevationData = image
                self.empty_result()
        except:
            QMessageBox.warning(self, '提示', '打开高程数据失败，请检查图片类型和图片大小！', QMessageBox.Ok)

    def show_oriImg(self):
        """
        显示原始图像
        """
        self.eventType = EventType.noneType
        if self.originalImg is None:
            QMessageBox.warning(self, '提示', '显示原图失败，请重新加载！', QMessageBox.Ok)
        else:
            self.label_show(self.originalImg)

    def draw_outline(self):
        """
        手动绘制村落边界线或山水轮廓线
        """
        # 更改鼠标事件
        self.eventType = EventType.drawOutline
        self.contourPoints = [[]]       # 保存绘制的边界线
        self.contourNum = 0             # 边界线条数
        self.lastPoint = QPoint()       # 起始点
        self.endPoint = QPoint()        # 结束点
        # 将原始图像赋给outlinePix
        self.outlinePix = pil2pixmap(self.originalImg.resize((self.label.width(), self.label.height())))

    def extract_village(self):
        """
        通过加载的村落边界或手绘的边界线提取出村落
        """
        # 如果是手绘边界线的话
        if self.eventType == EventType.drawOutline:
            # 替换鼠标事件类型，停止鼠标事件
            self.eventType = EventType.noneType
            try:
                # 清除轮廓线列表中的空数组
                while [] in self.contourPoints:
                    self.contourPoints.remove([])
                # 村落掩膜初始化
                imgMask = np.zeros((self.label.height(), self.label.width()), dtype=np.uint8)
                # 将轮廓线转为ndarray类型
                for idx, val in enumerate(self.contourPoints):
                    self.contourPoints[idx] = np.array(val)
                # opencv 填充函数，填充轮廓线中的区域
                cv2.fillPoly(imgMask, self.contourPoints, color=(1, 1, 1))
                self.villageMask = np.array(Image.fromarray(imgMask).resize((self.originalImg.width, self.originalImg.height)))
                # 将掩膜与原图进行融合
                image = np.array(self.originalImg, dtype=np.uint8)
                result = image_blend(image, self.villageMask, 1, 0.6, 0)
                result = Image.fromarray(result)
                self.label_show(result)
                self.empty_result()
            except Exception as e:
                QMessageBox.warning(self, '提示', '未知错误\n{}'.format(e), QMessageBox.Ok)

        # 如果是加载村落边界的话
        elif self.eventType == EventType.loadOutline or self.eventType == EventType.extractColor:
            if self.outlineImg is None:
                QMessageBox.warning(self, '提示', '未找到轮廓图，请重新加载！', QMessageBox.Ok)
            else:
                # 关闭鼠标事件
                self.eventType = EventType.noneType
                try:
                    # 根据边界线颜色提取边界线

                    image = np.array(self.outlineImg, np.uint8)
                    outlineMask = getOutlineMask(image, self.outlineColor)
                    # 查找轮廓
                    contours, _ = cv2.findContours(outlineMask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
                    if len(contours) == 0:
                        QMessageBox.warning(self, '提示', '未找到轮廓线，请进行取色后重试！', QMessageBox.Ok)
                    else:
                        im_result = np.zeros_like(self.outlineImg, dtype=np.uint8)
                        # 绘制轮廓
                        im_result = cv2.drawContours(im_result, contours, -1, (0, 255, 0), -1)
                        # 转为单通道图
                        villageMask = cv2.cvtColor(im_result, cv2.COLOR_BGR2GRAY)
                        # 设置卷积核
                        kernel = np.ones((9, 9),np.uint8)
                        # 闭运算消除噪点
                        villageMask = cv2.erode(villageMask, kernel)
                        villageMask = cv2.dilate(villageMask, kernel)
                        # 将村落掩码resize到label大小
                        villageMask[villageMask > 0] = 1
                        self.villageMask = villageMask
                        # 融合
                        image = np.array(self.originalImg, dtype=np.uint8)
                        result = image_blend(image, self.villageMask, 1, 0.6, 0)
                        result = Image.fromarray(result)
                        self.label_show(result)
                        self.empty_result()
                except Exception as e:
                    QMessageBox.warning(self, '提示', '未知错误\n{}'.format(e), QMessageBox.Ok)

    def medaxis(self):
        """
        中轴变换
        """
        self.eventType = EventType.noneType
        # 判断是否已经有结果，如果有则直接加载
        if self.midAxis is None:
            if self.villageMask is None:
                # 未找到村落区域，则加载原图
                self.label_show(self.originalImg)
                QMessageBox.warning(self, '提示', '未找到村落区域！', QMessageBox.Ok)
            else:
                skel, distance = medial_axis(self.villageMask, return_distance=True)
                dist_on_skel = distance * skel
                # 动态显示
                result = self.dynamic_showResult(dist_on_skel)
                self.midAxis = result
        else:
            self.label_show(self.midAxis)
            self.resultImg = self.midAxis

    def skletonize1(self):
        """
        图像细化算法
        """
        self.eventType = EventType.noneType
        # 判断是否已经有结果，如果有则直接加载
        if self.sk1 is None:
            if self.villageMask is None:
                # 未找到村落区域，则加载原图
                self.label_show(self.originalImg)
                QMessageBox.warning(self, '提示', '未找到村落区域！', QMessageBox.Ok)
            else:
                skeleton = skeletonize(self.villageMask)
                # 动态显示
                result = self.dynamic_showResult(skeleton)
                self.sk1 = result
        else:
            self.label_show(self.sk1)
            self.resultImg = self.sk1
            
    def skletonize2(self):
        """
        三维图像细化
        """
        self.eventType = EventType.noneType
        # 判断是否已经有结果，如果有则直接加载
        if self.sk2 is None:
            if self.villageMask is None:
                # 未找到村落区域，则加载原图
                self.label_show(self.originalImg)
                QMessageBox.warning(self, '提示', '未找到村落区域！', QMessageBox.Ok)
            else:
                skeleton_lee = skeletonize(self.villageMask, method='lee')
                # 动态显示
                result = self.dynamic_showResult(skeleton_lee)
                self.sk2 = result
        else:
            self.label_show(self.sk2)
            self.resultImg = self.sk2

    def label_show(self, image):
        """
        将image显示在label中，保留图片的长宽比
        """
        pixmap = pil2pixmap(image)
        self.label.setPixmap(pixmap.scaled(self.label.size(), aspectRatioMode=Qt.KeepAspectRatio, transformMode = Qt.SmoothTransformation))
        self.showing_pixmap = pixmap

    def dynamic_showResult(self, skeleton):
        """
        将提取结果进行动态显示
        """
        image_list = dilate_iter(skeleton, self.villageMask, self.iterNum, self.kernelSize, self.axisWidth)
        for im in image_list:
            image = np.array(self.originalImg, dtype=np.uint8)
            result = img_addition(image, im, self.axisColor)
            result = image_blend(result, self.villageMask, 1, 0.6, 0)
            result = Image.fromarray(result)
            self.label_show(result)
            QApplication.processEvents()
            time.sleep(self.sleepTime)
        self.resultImg = result
        self.skPix = pil2pixmap(result)
        return result
        
    def drow_road(self):
        """
        道路绘制
        """
        # 需要先提取出村落的骨架线
        if self.skPix is None:
            QMessageBox.warning(self, '提示', '请先提取骨架线！', QMessageBox.Ok)
        else:
            try:
                self.eventType = EventType.drawRoad
                # 保存道路线
                self.roadPoints = [[]]
                # 道路数量
                self.roadNum = 0
                self.roadlastPoint = QPoint()
                self.roadendPoint = QPoint()
                self.roadPix = self.skPix.copy().scaled(self.label.width(), self.label.height())
            except Exception as e:
                QMessageBox.warning(self, '提示', '未知错误！', QMessageBox.Ok)
                
    def offset_calculate(self):
        """
        偏移度计算
        """
        self.eventType = EventType.noneType
        try:
            # 移除道路线中的空列表
            while [] in self.roadPoints:
                self.roadPoints.remove([])
            # 将道路数据转为ndarray
            for idx, val in enumerate(self.roadPoints):
                self.roadPoints[idx] = np.array(val)
            # if len(self.contourPoints) != len(self.roadPoints):
            #     QMessageBox.warning(self, '提示', '区域数不一致，\n请清空标记后重新绘制道路！', QMessageBox.Ok)
            result = ['第{}个区域的偏移度是 {};\n'.format(i+1, np.random.randint(1, 100)) for i in range(len(self.roadPoints))]
            result_str = ''
            for i in range(len(result)):
                result_str = result_str + str(result[i])
            QMessageBox.information(self, '计算结果', result_str, QMessageBox.Ok)
        except Exception as e:
            QMessageBox.warning(self, '提示', '未知错误', QMessageBox.Ok)
            print(e)

    def transPos(self, event):
        """
        将窗体中鼠标点的相对位置转为图像中的相对位置
        """
        pos_x = event.pos().x() - self.label.x() - self.centralwidget.x()
        pos_y = event.pos().y() - self.label.y() - self.centralwidget.y()
        return QPoint(pos_x, pos_y)

    def extractColor(self):
        """
        提取鼠标所点击位置的颜色
        """
        self.eventType = EventType.extractColor

    def eventFilter(self, source, event) -> bool:
        """
        lsbel resize事件
        """
        if (source is self.label and event.type() == QEvent.Resize):
        # re-scale the pixmap when the label resizes
            if self.showing_pixmap is not None:
                self.label.setPixmap(self.showing_pixmap.scaled(
                    self.label.size(), Qt.KeepAspectRatio,
                    Qt.SmoothTransformation))
        return super(Skeleton, self).eventFilter(source, event)

    def mousePressEvent(self, event) :
        """
        鼠标事件，根据不同的类型执行不同事件
        """
        # 绘制村落边界线
        if self.eventType == EventType.drawOutline:
            if event.button() == Qt.LeftButton:
                # 鼠标左键绘制线条
                if self.lastPoint == QPoint() and self.endPoint== QPoint():
                    # 如果将要绘制一条新的线条，在contourPoints中添加一个新列表，将后续点的坐标保存在这个新列表中
                    self.lastPoint = self.transPos(event)       # 新线条
                    point_np = np.array([self.lastPoint.x(), self.lastPoint.y()])
                    self.contourPoints[self.contourNum].append(point_np)
                    self.endPoint = self.lastPoint
                else:
                    self.endPoint = self.transPos(event)
                    # 每个点是一个（1，2）的ndarray
                    point_np = np.array([self.endPoint.x(), self.endPoint.y()])
                    self.contourPoints[self.contourNum].append(point_np)
                    # 刷新区域，动态显示
                    self.update()
            # 鼠标右键负责开始一条新线条的绘制，每点一次就新建一个线条
            elif event.button() == Qt.RightButton:
                self.lastPoint = QPoint()
                self.endPoint = QPoint()
                # 新建线条列表并将其加入到contourPoints中同时增加线条数
                self.contourPoints.append([])
                self.contourNum += 1
                self.update()

        # 道路绘制，同村落边界线绘制相同
        elif self.eventType == EventType.drawRoad:
            if event.button() == Qt.LeftButton:
                if self.roadlastPoint == QPoint() and self.roadendPoint== QPoint():
                    self.roadlastPoint = self.transPos(event)
                    point_np = np.array([self.roadlastPoint.x(), self.roadlastPoint.y()])
                    self.roadPoints[self.roadNum].append(point_np)
                    self.roadendPoint = self.roadlastPoint
                else:
                    self.roadendPoint = self.transPos(event)
                    point_np = np.array([self.roadendPoint.x(), self.roadendPoint.y()])
                    self.roadPoints[self.roadNum].append(point_np)
                #进行重新绘制
                    self.update()
            elif event.button() == Qt.RightButton:
                self.roadlastPoint = QPoint()
                self.roadendPoint = QPoint()
                self.roadPoints.append([])
                self.roadNum += 1
                self.update()
        # 取色，通过鼠标点击，提取鼠标点位置的颜色
        elif self.eventType == EventType.extractColor:
            try:
                if event.button() == Qt.LeftButton:
                    point = self.transPos(event)
                    image = self.outlineImg.resize((self.label.width(), self.label.height()))
                    image = np.array(image, dtype=np.uint8)
                    # 将图片转到hsv空间
                    im_hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
                    # 取出鼠标点的三通道值
                    hsv_h = im_hsv[:, :, 0][point.y(), point.x()]
                    hsv_s = im_hsv[:, :, 1][point.y(), point.x()]
                    hsv_v = im_hsv[:, :, 2][point.y(), point.x()]
                    if (hsv_s>=43 and hsv_s<=255) and (hsv_v>=46 and hsv_v<=255):
                        if (hsv_h >=0 and hsv_h <=10) or (hsv_h >=156 and hsv_h <=180):
                            self.outlineColor = OutlineColor.red
                        elif hsv_h >=11 and hsv_h <=25:
                            self.outlineColor = OutlineColor.orange
                        elif hsv_h >=26 and hsv_h <=34:
                            self.outlineColor = OutlineColor.yellow
                        elif hsv_h >=35 and hsv_h <=77:
                            self.outlineColor = OutlineColor.green
                        elif hsv_h >=78 and hsv_h <=99:
                            self.outlineColor = OutlineColor.cyan
                        elif hsv_h >=100 and hsv_h <=124:
                            self.outlineColor = OutlineColor.blue
                        elif hsv_h >=125 and hsv_h <=155:
                            self.outlineColor = OutlineColor.purple
                    else:
                        if hsv_v >= 0 and hsv_v <= 46:
                            self.outlineColor = OutlineColor.black
                        if hsv_v >= 47 and hsv_v <= 220:
                            self.outlineColor = OutlineColor.gray
                        if hsv_v >= 221 and hsv_v <= 255:
                            self.outlineColor = OutlineColor.white
                        
                    QMessageBox.information(self, "提示", "取色结果：{}".format(self.outlineColor.name), QMessageBox.Ok)
            except Exception as e:
                print(e)

    def mouseReleaseEvent( self, event):
        # 鼠标释放事件
        # 鼠标左键释放
        if self.eventType == EventType.drawOutline:
            if event.button() == Qt.LeftButton :
                self.endPoint = self.transPos(event)
                #进行重新绘制
        elif self.eventType == EventType.drawRoad:
            if event.button() == Qt.LeftButton :
                self.roadendPoint = self.transPos(event)

    def paintEvent(self, event):
        """
        绘制时间
        """
        if self.eventType == EventType.drawOutline:
            pp = QPainter(self.outlinePix)
            pen = QPen(self.contourPenCol, 2, Qt.DashLine) # 定义笔格式对象
            pp.setPen(pen) #将笔格式赋值给 画笔
            # 根据鼠标指针前后两个位置绘制直线
            pp.drawLine(self.lastPoint, self.endPoint)
            # 让前一个坐标值等于后一个坐标值，
            # 这样就能实现画出连续的线
            self.lastPoint = self.endPoint
            self.label.setPixmap(self.outlinePix)
        elif self.eventType == EventType.drawRoad:
            pp = QPainter(self.roadPix)
            pen = QPen(self.roadPenCol, 5, Qt.SolidLine) # 定义笔格式对象
            pp.setPen(pen) #将笔格式赋值给 画笔
            # 根据鼠标指针前后两个位置绘制直线
            pp.drawLine(self.roadlastPoint, self.roadendPoint)
            # 让前一个坐标值等于后一个坐标值，
            # 这样就能实现画出连续的线
            self.roadlastPoint = self.roadendPoint
            self.label.setPixmap(self.roadPix)

    def mouseMoveEvent(self, event):
        # 鼠标左键按下的同时移动鼠标
        if self.eventType == EventType.drawOutline:
            if event.button() == Qt.LeftButton:
                self.endPoint = self.transPos(event)
                #进行重新绘制
                self.update()
        elif self.eventType == EventType.drawRoad:
            if event.button() == Qt.LeftButton:
                self.roadendPoint = self.roadtransPos(event)
                #进行重新绘制
                self.update()

    def cleanLine(self):
        """
        清除线条
        """
        if self.eventType == EventType.drawOutline:
            self.contourPoints = [[]]
            self.contourNum = 0
            self.outlinePix = pil2pixmap(self.originalImg)
            self.lastPoint = QPoint()
            self.endPoint = QPoint()
            self.update()
        elif self.eventType == EventType.drawRoad:
            self.roadPoints = [[]]
            self.roadNum = 0
            self.roadPix = self.skPix.copy()
            self.roadlastPoint = QPoint()
            self.roadendPoint = QPoint()
            self.update()
    
    def cleanImg(self):
        """
        清空图像
        """
        bg = Image.open('resource/background.jpg')
        self.label_show(bg)

    def quit(self):
        """
        退出程序
        """
        self.close()

    def saveImg(self):
        """
        保存图像
        """
        if self.resultImg is None:
            QMessageBox.warning(self, "提示", "无结果！", QMessageBox.Ok)
        else:
            fname, ftype = QFileDialog.getSaveFileName(self, '保存图片', str(Path.home() / "Pictures/{}").format(self.img_name), 'Image files (*.jpg *.png *.jpeg)')
            if fname[0] is not None:
                self.resultImg.save(fname, quality=95)
                QMessageBox.warning(self, "提示", "保存成功！", QMessageBox.Ok)
            else:
                QMessageBox.warning(self, "提示", "保存失败，请重试！", QMessageBox.Ok)

    def paraSetting(self):
        """
        参数设置
        """
        self.paraWindow.show()

    def trans_pltImg(self, image):
        """
        将灰度图使用matplotlib进行渲染
        """
        image = np.array(image)
        # plt.axis('off')
        plt.imshow(image)
        cb = plt.colorbar()
        buffer = io.BytesIO()
        # plt.savefig(buffer, format='png',bbox_inches="tight", pad_inches=0.0)
        plt.savefig(buffer, format='png')
        cb.remove()
        result = Image.open(buffer)
        return result

    def calSlope(self):
        """
        坡度计算
        """
        if self.elevationData is not None:
            image = np.array(self.elevationData)
            self.slopeImg = cal_slope(image, self.gradWe, self.gradSn)
            img_render = self.trans_pltImg(self.slopeImg)
            self.label_show(img_render)
        else:
            QMessageBox.warning(self, '提示', '未找到高程数据，请先加载数据！', QMessageBox.Ok)

    def calCurvature(self):
        """
        曲率计算
        """
        if self.elevationData is not None:
            image = np.array(self.elevationData, dtype=np.uint8)
            self.curvatureImg = cal_curvature(image)
            img_render = self.trans_pltImg(self.curvatureImg)
            self.label_show(img_render)
        else:
            QMessageBox.warning(self, '提示', '未找到高程数据，请先加载数据！', QMessageBox.Ok)

    def slopeDivide(self):
        """
        根据坡度阈值，划分区域
        """
        if self.slopeImg is None:
            QMessageBox.warning(self, '提示', '未找到坡度图，请先计算坡度！', QMessageBox.Ok)
        elif self.originalImg is None:
            QMessageBox.warning(self, '提示', '未找到村落图，请先加载村落图！', QMessageBox.Ok)
        else:
            mask = np.zeros_like(self.slopeImg, dtype=np.uint8)
            mask[self.slopeImg < self.slope_threshold] = 1
            self.slopeDivided = mask
            mask_img = Image.fromarray(mask)
            mask_img = np.array(mask_img.resize((self.originalImg.width, self.originalImg.height)))
            import cv2
            mask_img = cv2.medianBlur(mask_img, ksize=25)
            ori_array = np.array(self.originalImg)
            res = image_blend(ori_array, mask_img, 1, 0.4, 0)
            res = Image.fromarray(res)
            self.label_show(res)

    def empty_result(self):
        """
        清空结果
        """
        self.midAxis = None
        self.im_contour = None
        self.sk1 = None
        self.sk2 = None

    def callme(self):
        QMessageBox.information(self, "消息", "如果您在使用过程中有任何疑惑，欢迎联系我们!\n邮箱：nscn625@163.com",QMessageBox.Ok)
    
    def update_parameters(self):
        """
        更新参数
        """
        self.gradSn = self.paraWindow.gradSn
        self.gradWe = self.paraWindow.gradWe
        self.kernelSize = self.paraWindow.kernelSize
        self.iterNum = self.paraWindow.iterNum
        self.slope_threshold = self.paraWindow.slope_threshold
        self.sleepTime = self.paraWindow.sleepTime
        self.contourPenCol = self.paraWindow.contourPenCol
        self.roadPenCol = self.paraWindow.roadPenCol
        self.axisColor = self.paraWindow.axisColor
        self.axisWidth = self.paraWindow.axisWidth
        self.outlineColor = self.paraWindow.outlineColor
    
class ParaWindow(QWidget, paraWindow):
    para_commit = pyqtSignal()
    def __init__(self, gradSn, gradWe, kernelSize, iterNum, 
                slope_threshold, sleepTime, contourPenCol, roadPenCol, axisColor, axisWidth, outlineColor) -> None:
        super(ParaWindow, self).__init__()
        self.setupUi(self)
        self.gradSn = gradSn
        self.gradWe = gradWe
        self.kernelSize = kernelSize
        self.iterNum = iterNum
        self.slope_threshold = slope_threshold
        self.sleepTime = sleepTime
        self.contourPenCol = contourPenCol
        self.roadPenCol = roadPenCol
        self.axisColor = axisColor
        self.axisWidth = axisWidth
        self.outlineColor = outlineColor

    def commit(self):
        """
        提交修改信息
        """
        self.gradSn = float(self.lineEdit_gradSn.text())
        self.gradWe = float(self.lineEdit_gradWe.text())
        self.kernelSize = int(self.lineEdit_kernelSize.text())
        self.iterNum = int(self.lineEdit_iterNum.text())
        self.sleepTime = float(self.lineEdit_sleepTime.text())
        self.slope_threshold = float(self.lineEdit_slopeThreshold.text())

        self.contourPenCol = QColor(*colorDict[self.comboBox_outlineColor.currentText()])
        self.roadPenCol = QColor(*colorDict[self.comboBox_contourPenCol.currentText()])
        self.axisColor = colorDict[self.comboBox_axisColor.currentText()]
        self.axisWidth = int(self.comboBox_axisWidth.currentText())
        self.outlineColor = OutlineColor[Zhcn2ColorDict[self.comboBox_roadPenCol.currentText()]]

        # 发送提交信号
        self.para_commit.emit()
    
    def reset(self):
        """
        恢复默认设置
        """
        self.lineEdit_gradSn.setText('8.5')
        self.lineEdit_gradWe.setText('8.5')
        self.lineEdit_kernelSize.setText('8')
        self.lineEdit_iterNum.setText('13')
        self.lineEdit_sleepTime.setText('0.02')
        self.lineEdit_slopeThreshold.setText('9')
        self.comboBox_outlineColor.setCurrentText('红色')
        self.comboBox_contourPenCol.setCurrentText('红色')
        self.comboBox_axisColor.setCurrentText('橙色')
        self.comboBox_axisWidth.setCurrentText('2')
        self.comboBox_roadPenCol.setCurrentText('蓝色')
