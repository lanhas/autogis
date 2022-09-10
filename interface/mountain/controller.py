from PyQt5.QtCore import pyqtSignal
from PyQt5.QtWidgets import QWidget
from mountain import Ui_MainWindow as mountainWindow
from parameters import Ui_Form as paraWindow
from func import *

class MountainExtract(BaseMainWindow, mountainWindow):
    def __init__(self, parent=None) -> None:
        super(MountainExtract, self).__init__(parent=parent)
        self.setupUi(self)
        # 初始化
        self.originalImg = None     # 加载的原始图片，Image类型
        self.originalDem = None     # 加载的高程数据， Image类型
        self.resultMask = None      # 得到的掩码二值结果, ndarray
        self.resultImg = None       # 得到的用于显示的最终结果

        # 默认参数
        self.extract_mode = 'simple'
        self.adjust_mode = 'simple'
        self.grad_we = 0.53     # dem格网宽度，0.53米/像素
        self.grad_sn = 0.53     # dem格网高度，0.53米/像素
        self.kernel_close = 30  # 闭运算核大小，用于消除狭长的山地
        self.close_iter = 2     # 闭运算迭代次数
        self.kernel_open = 30   # 开运算核大小，用于消除细长的平原
        self.open_iter = 1      # 开运算迭代次数
        self.division_coef = 0.25       # 高程分界线 = m_eventdot +division_coef*(m_maxdot-m_eventdot)
        self.slope_threshold = 9       # 坡度阈值
        # area = 30000*0.53*0.53 = 8427 (m2) 
        self.plain_threshold = 30000    # 山区调整时平原的面积阈值，小于该面积的会被删去
        self.hill_threshold = 30000     # 山区调整时小山的面积阈值， 小于该面积的会被删去

        # 参数窗体初始化
        self.paraWindow = ParaWindow(self.grad_sn, self.grad_we, self.kernel_close, self.kernel_open, self.open_iter,self.close_iter, 
                                    self.plain_threshold, self.hill_threshold, self.division_coef, self.slope_threshold, self.extract_mode, self.adjust_mode)
        self.paraWindow.para_commit.connect(self.update_parameters)

    def openImg(self):
        """
        加载遥感数据
        """
        try:
            fname ,_ = QFileDialog.getOpenFileName(self,'Open File','interface/mountain_foot/data',
                                                    'Image files (*.jpg *.tif *.tiff *.png *.jpeg)')
            img = Image.open(fname)
            self.originalImg = img
            self.label.setPixmap(pil2pixmap(img.resize((self.label.width(), self.label.height()))))
        except Exception as e:
            QMessageBox.warning(self, '提示', '打开图片失败，请检查图片类型和图片大小！', QMessageBox.Ok)

    def openDem(self):
        """
        加载高程数据
        """
        try:
            fname ,_ = QFileDialog.getOpenFileName(self,'Open File','interface/mountain_foot/data',
                                                    'Dem files (*.jpg *.tif *.tiff *.png *.jpeg)')
            image = Image.open(fname)
            self.originalDem = image.copy()
            result = tif2bmp(image)
            self.label.setPixmap(pil2pixmap(result.resize((self.label.width(), self.label.height()))))
        except Exception as e:
            QMessageBox.warning(self, '提示', '打开图片失败，请检查高程数据类型和大小！', QMessageBox.Ok)

    def originalImg(self):
        """
        显示原始的遥感数据
        """
        if self.originalImg is None:
            QMessageBox.warning(self, '提示', '请先加载图片！', QMessageBox.Ok)
        else:
            self.label.setPixmap(pil2pixmap(self.originalImg.resize((self.label.width(), self.label.height()))))

    def originalDem(self):
        """
        显示原始的高程数据
        """
        if self.originalDem is None:
            QMessageBox.warning(self, '提示', '请先加载图片！', QMessageBox.Ok)
        else:
            image = self.originalDem.copy()
            result = tif2bmp(image)
            self.label.setPixmap(pil2pixmap(result.resize((self.label.width(), self.label.height()))))

    def extract(self):
        """
        提取山区，将图片分为山区和平原
        """
        try:
            if self.originalDem is None:
                QMessageBox.warning(self, '提示', '请先加载高程数据！', QMessageBox.Ok)
            else:
                image = np.array(self.originalDem)
                # 提取出山区掩膜
                image = mountain_extract(image, self.grad_we, self.grad_sn, self.division_coef, 
                                                    slope_threshold=self.slope_threshold, mode=self.extract_mode)
                # 山区调整
                image = mountain_adjust(image, self.kernel_close, self.close_iter, self.kernel_open, 
                                                    self.open_iter, self.plain_threshold, self.hill_threshold, mode=self.adjust_mode)
                image[image==1] = 255
                self.resultMask = image
                result = Image.fromarray(self.resultMask)
                self.label.setPixmap(pil2pixmap(result.resize((self.label.width(), self.label.height()))))
        except Exception as e:
            QMessageBox.warning(self, "警告", "预测失败，请重试！\n 错误：{}".format(e), QMessageBox.Ok)

    def showResult(self):
        """
        将提取的山区、平原掩膜与原图像融合
        """
        if self.resultMask is None:
            QMessageBox.warning(self, '提示', '请先进行预测！', QMessageBox.Ok)
        else:
            image = np.array(self.originalImg, dtype=np.uint8)
            areaMask = self.resultMask

            image = image_blend(image, areaMask, 1, 0.4, 0)
            self.result = Image.fromarray(image)
            self.label.setPixmap(pil2pixmap(self.result.resize((self.label.width(), self.label.height()))))
    
    def calculate(self):
        """
        计算区域的高程信息
        """
        image = np.array(self.originalDem)
        m_max = image.max()
        m_min = image.min()
        m_mean = image.mean()
        QMessageBox.information(self, "高程信息", "区域最大高程：{}\n 区域最小高程：{}\n 区域平均高程：{}\n".format(m_max, m_min, m_mean), QMessageBox.Ok)
     
    def paraSetting(self):
        """
        参数设置
        """
        self.paraWindow.show()
        
    def show_slope(self):
        """
        显示坡度图
        """
        pass

    def show_curvKh(self):
        """
        显示平面曲率
        """
        pass

    def show_curvKv(self):
        """
        显示剖面曲率
        """
        pass

    def drawMountainFoot(self):
        """
        绘制山脚线
        """
        mask = adjust_mask(self.resultMask)
        temp = Image.fromarray(mask)
        temp.show()
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        image = np.zeros_like(self.originalImg, dtype=np.uint8)
        image = cv2.drawContours(image, contours, 0, (0, 255, 0), -1)
        result = Image.fromarray(image)
        self.label.setPixmap(pil2pixmap(result.resize((self.label.width(), self.label.height()))))

    def cleanImg(self):
        bg = Image.open('interface/axis_trans/resource/background.jpg')
        self.label.setPixmap(pil2pixmap(bg))
        self.empty_result()

    def quit(self):
        pass

    def setSingleImg(self):
        pass

    def setFolderImg(self):
        pass

    def saveImg(self):
        if self.resultImg is None:
            QMessageBox.warning(self, "提示", "无结果！", QMessageBox.Ok)
        else:
            fname, ftype = QFileDialog.getSaveFileName(self, '保存图片', 'interface/mountain_foot/result', 'Image files (*.jpg *.png *.jpeg)')
            if fname[0] is not None:
                self.resultImg.save(fname, quality=95)
                QMessageBox.warning(self, "提示", "保存成功！", QMessageBox.Ok)
            else:
                QMessageBox.warning(self, "提示", "保存失败，请重试！", QMessageBox.Ok)

    def startProgressBar(self):
        self.qb = ProgressBar()
        self.qb.show()
        self.qb.onStart()

    def empty_result(self):
        """
        清空结果缓存
        """
        self.originalImg = None     # 加载的原始图片，Image类型
        self.originalDem = None     # 加载的高程数据， Image类型
        self.resultMask = None      # 得到的掩码二值结果, ndarray
        self.resultImg = None       # 得到的用于显示的最终结果

    def update_parameters(self):
        """
        更新参数
        """
        self.extract_mode = self.paraWindow.excateMode
        self.adjust_mode = self.paraWindow.adjustMode
        self.grad_we = self.paraWindow.gradWe
        self.grad_sn = self.paraWindow.gradSn
        self.kernel_close = self.paraWindow.kernelClose
        self.close_iter = self.paraWindow.iterClose
        self.kernel_open = self.paraWindow.kernelOpen
        self.open_iter = self.paraWindow.iterOpen
        self.division_coef = self.paraWindow.divisionCoef
        self.slope_threshold = self.paraWindow.slopeThreshold
        self.plain_threshold = self.paraWindow.plainThreshold
        self.hill_threshold = self.paraWindow.hillThreshold

class ParaWindow(QWidget, paraWindow):
    para_commit = pyqtSignal()
    def __init__(self, gradSn, gradWe, kernelClose, kernelOpen, 
                iterClose, iterOpen, plainThreshold, hillThreshold, divisionCoef, slopeThreshold, excateMode, adjustMode) -> None:
        super(ParaWindow, self).__init__()
        self.setupUi(self)
        self.gradSn = gradSn
        self.gradWe = gradWe
        self.kernelClose = kernelClose
        self.kernelOpen = kernelOpen
        self.iterClose = iterClose
        self.iterOpen = iterOpen
        self.plainThreshold = plainThreshold
        self.hillThreshold = hillThreshold
        self.divisionCoef = divisionCoef
        self.slopeThreshold = slopeThreshold
        self.excateMode = excateMode
        self.adjustMode = adjustMode

    def reset(self):
        """
        恢复默认设置
        """
        self.lineEdit_gradSn.setText('0.53')
        self.lineEdit_gradWe.setText('0.53')
        self.lineEdit_kernelClose.setText('30')
        self.lineEdit_kernelOpen.setText('30')
        self.lineEdit_iterClose.setText('2')
        self.lineEdit_iterOpen.setText('1')
        self.lineEdit_plainThreshold.setText('30000')
        self.lineEdit_hillThreshold.setText('30000')
        self.lineEdit_divisionCoef.setText('0.25')
        self.lineEdit_slopeThreshold.setText('9')
        self.comboBox_excateMode.setCurrentText('simple')
        self.comboBox_adjuseMode.setCurrentText('simple')

    def commit(self):
        """
        提交修改信息
        """
        self.gradSn = float(self.lineEdit_gradSn.text())
        self.gradWe = float(self.lineEdit_gradWe.text())
        self.kernelClose = int(self.lineEdit_kernelClose.text())
        self.kernelOpen = int(self.lineEdit_kernelOpen.text())
        self.iterClose = int(self.lineEdit_iterClose.text())
        self.iterOpen = int(self.lineEdit_iterOpen.text())
        self.plainThreshold = int(self.lineEdit_plainThreshold.text())
        self.hillThreshold = int(self.lineEdit_hillThreshold.text())
        self.divisionCoef = float(self.lineEdit_divisionCoef.text())
        self.slopeThreshold = float(self.lineEdit_slopeThreshold.text())
        self.excateMode = self.comboBox_excateMode.currentText()
        self.adjustMode = self.comboBox_adjuseMode.currentText()
        # 发送提交信号
        self.para_commit.emit()

class ProgressBar(QtWidgets.QWidget):
    def __init__(self, parent= None):
        QtWidgets.QWidget.__init__(self)
        
        self.setGeometry(300, 300, 250, 150)
        self.setWindowTitle('ProgressBar')
        self.pbar = QtWidgets.QProgressBar(self)
        self.pbar.setGeometry(30, 40, 200, 25)
        
        self.button = QtWidgets.QPushButton('Start', self)
        self.button.setFocusPolicy(Qt.NoFocus)
        self.button.move(40, 80)
        
        self.button.clicked.connect(self.onStart)
        self.timer = QBasicTimer()
        self.step = 0
        
    def timerEvent(self, event):
        if self.step >=100:
            self.timer.stop()
            return
        self.step = self.step + 1
        self.pbar.setValue(self.step)
        
    def onStart(self):
        if self.timer.isActive(): 
            self.timer.stop()
            self.button.setText('Start')
        else:
            self.timer.start(100, self)
            self.button.setText('Stop')

    def quit(self):
        self.close()