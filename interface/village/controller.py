from func import *
from PyQt5.QtWidgets import QWidget
from PyQt5.QtCore import pyqtSignal
from parameters import Ui_Form as paraWindow
from village import Ui_MainWindow as villageWindow
from predict.predict_mtvc import test

class VillageClassification(BaseMainWindow, villageWindow):
    def __init__(self, parent=None) -> None:
        super(VillageClassification, self).__init__(parent=parent)
        self.setupUi(self)
        # 初始化
        self.originalImg = None     # 遥感图像，Image类型
        self.originalDem = None     # 高程数据，Image类型
        self.resultMask = None      # 预测结果，Image类型
        self.resultImg = None       # 融合结果，Image类型
        self.model_SPath = None     # 分割模型地址
        self.model_CPath = None     # 分类模型地址

        self.pmode = 'M'
        self.backbone = 'resnet50'
        self.device = 'cuda'
        self.pmethod = 'normal'
        self.resize = -1
        self.num_class = 7
        self.mtss_predicter = MtssPredicter(self.pmode, self.backbone, self.device, self.pmethod, self.resize, self.num_class)  # 预测类
        self.mtvc_predicter = MtvcPredicter(self.device, self.resize, 5)
        self.paraWindow = ParaWindow(self.pmode, self.backbone, self.device, self.resize, self.pmethod, self.num_class)
        self.paraWindow.para_commit.connect(self.update_parameters)

    def openImg(self):
        try:
            fname ,_ = QFileDialog.getOpenFileName(self,'Open File','interface/village_classification/data/remote',
                                                    'Image files (*.jpg *.tif *.tiff *.png *.jpeg)')
            if fname != '':
                self.originalImg = Image.open(fname)
                self.label_show(self.originalImg)
        except Exception as e:
            QMessageBox.warning(self, '提示', '打开图片失败，请检查图片类型和图片大小！', QMessageBox.Ok)

    def openDem(self):
        try:
            fname ,_ = QFileDialog.getOpenFileName(self,'Open File','interface/village_classification/data/dem',
                                                    'Image files (*.jpg *.tif *.tiff *.png *.jpeg)')
            if fname != '':
                self.originalDem = Image.open(fname)
                self.label_show(self.originalDem)
        except Exception as e:
            QMessageBox.warning(self, '提示', '打开图片失败，请检查图片类型和图片大小！', QMessageBox.Ok)

    def load_Mask(self):
        try:
            fname ,_ = QFileDialog.getOpenFileName(self,'Open File','datasets/mtvcd/villageMask',
                                                    'Image files (*.jpg *.tif *.tiff *.png *.jpeg)')
            if fname != '':
                self.resultMask = Image.open(fname)
                if self.resultMask.mode == 'P':
                    image = decode_fn(self.resultMask)
                else:
                    image = self.resultMask
                self.label_show(image)
        except Exception as e:
            QMessageBox.warning(self, '提示', '打开图片失败，请检查图片类型和图片大小！', QMessageBox.Ok)

    def openDir(self):
        pass

    def load_SModel(self):
        fname ,_ = QFileDialog.getOpenFileName(self,'Open model','checkpoints/mtss',
                                                'model files (*.pth)')
        self.model_SPath = fname
        self.mtss_predicter.load_model(fname)

    def load_CModel(self):
        fname ,_ = QFileDialog.getOpenFileName(self,'Open model','checkpoints/mtvc',
                                                'model files (*.pth)')
        self.model_CPath = fname
        self.mtvc_predicter.load_model(fname)

    def show_remoteImg(self):
        """
        显示原始的遥感图像
        """
        if self.originalImg is None:
            QMessageBox.warning(self, '提示', '请先加载图片！', QMessageBox.Ok)
        else:
            self.label_show(self.originalImg)

    def mtss_predict(self):
        """
        村落地理要素分割
        """
        try:
            if self.model_SPath is None :
                QMessageBox.warning(self, '提示', '请先加载分割模型！', QMessageBox.Ok)
            else:
                self.resultMask = self.mtss_predicter.predict(self.originalImg, self.originalDem)
                # 上色
                result = decode_fn(self.resultMask).astype('uint8')
                self.label_show(result)
        except Exception as e:
            QMessageBox.warning(self, "警告", "预测失败，请重试！\n 错误：{}".format(e), QMessageBox.Ok)

    def mtvc_predict(self):
        try:
            if self.model_CPath is None :
                QMessageBox.warning(self, '提示', '请先分类加载模型！', QMessageBox.Ok)
            if self.resultMask is None:
                QMessageBox.warning(self, '提示', '请先得到特征图！', QMessageBox.Ok)
            else:
                self.mtvc_predicter.predict(self.resultMask)
        except Exception as e:
            QMessageBox.warning(self, "警告", "预测失败，请重试！\n 错误：{}".format(e), QMessageBox.Ok)

    def blendImg(self):
        """
        根据combbox所选类别，显示分类结果
        """
        if self.resultMask is None:
            QMessageBox.warning(self, '提示', '请先进行预测！', QMessageBox.Ok)
        if self.originalImg is None:
            QMessageBox.warning(self, '提示', '请先加载遥感图像！', QMessageBox.Ok)
        else:
            image = np.array(self.resultMask, dtype=np.uint8)
            areaMask = getAreaMask(image, self.comboBox.currentIndex())
            image = np.array(self.originalImg, dtype=np.uint8)
            result = image_blend(image, areaMask, 1, 0.6, 0)
            self.resultImg = Image.fromarray(result)
            self.label_show(self.resultImg)

    def display(self):
        test()

    def saveImg(self):
        """
        保存图像
        """
        if self.resultImg is None:
            QMessageBox.warning(self, "提示", "无结果！", QMessageBox.Ok)
        else:
            fname, ftype = QFileDialog.getSaveFileName(self, '保存图片', 'village_classification/result', 'Image files (*.jpg *.png *.jpeg)')
            if fname[0] is not None:
                self.resultImg.save(fname, quality=95)
                QMessageBox.warning(self, "提示", "保存成功！", QMessageBox.Ok)
            else:
                QMessageBox.warning(self, "提示", "保存失败，请重试！", QMessageBox.Ok)

    def cleanImg(self):
        bg = Image.open('axis_trans/resource/background.jpg')
        self.label.setPixmap(pil2pixmap(bg))
        self.empty_result()

    def label_show(self, image):
        """
        将image显示在label中，保留图片的长宽比
        """
        pixmap = pil2pixmap(image)
        self.label.setPixmap(pixmap.scaled(self.label.size(), aspectRatioMode=Qt.KeepAspectRatio, transformMode = Qt.SmoothTransformation))
        self.showing_pixmap = pixmap

    def quit(self):
        pass

    def empty_result(self):
        """
        清空缓存
        """
        self.originalImg = None     # 遥感图像，Image类型
        self.originalDem = None     # 高程数据，Image类型
        self.resultMask = None      # 预测结果，Image类型
        self.resultImg = None       # 融合结果，Image类型
        self.model_SPath = None     # 分割模型地址
        self.model_CPath = None     # 分类模型地址

    def update_parameters(self):
        """
        更新参数
        """
        self.pmode = self.paraWindow.pmode
        self.backbone = self.paraWindow.backbone
        self.device = self.paraWindow.device
        self.pmethod = self.paraWindow.pmethod
        self.num_class = self.paraWindow.num_class
        self.size = self.paraWindow.size

class ParaWindow(QWidget, paraWindow):
    para_commit = pyqtSignal()
    def __init__(self, pmode, backbone, device, size, pmethod, num_class) -> None:
        super(ParaWindow, self).__init__()
        self.setupUi(self)
        self.pmode = pmode
        self.backbone = backbone
        self.device = device
        self.size = size
        self.pmethod = pmethod
        self.num_class = num_class

    def commit(self):
        """
        提交修改信息
        """
        self.pmode = self.comboBox_pmode.text()
        self.backbone = self.comboBox_backbone.text()
        self.size = int(self.comboBox_resize.text())
        self.device = self.comboBox_device.text()
        self.pmethod = self.comboBox_pmethod.text()
        self.num_class = int(self.comboBox_numclass.text())
        # 发送提交信号
        self.para_commit.emit()
    
    def reset(self):
        """
        恢复默认设置
        """
        self.comboBox_pmode.setCurrentText('多模态')
        self.comboBox_backbone.setCurrentText('resnet50')
        self.comboBox_device.setCurrentText('cpu')
        self.comboBox_resize.setCurrentText('-1')
        self.comboBox_pmethod.setCurrentText('普通')
        self.comboBox_numclass.setCurrentText('7')





