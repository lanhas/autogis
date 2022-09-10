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