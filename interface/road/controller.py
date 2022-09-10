from func import *
from PyQt5.QtCore import pyqtSignal
from parameters import Ui_Form as paraWindow
from road import Ui_MainWindow as roadWindow

class RoadExtract(BaseMainWindow, roadWindow):
    def __init__(self, parent=None) -> None:
        super(RoadExtract, self).__init__(parent=parent)
        self.setupUi(self)
        # 初始化
