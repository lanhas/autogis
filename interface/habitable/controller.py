from func import *
from PyQt5.QtCore import pyqtSignal
from parameters import Ui_Form as paraWindow
from habitable import Ui_MainWindow as habitableWindow

class HabitablePred(BaseMainWindow, habitableWindow):
    def __init__(self, parent=None) -> None:
        super(HabitablePred, self).__init__(parent=parent)
        self.setupUi(self)
        # 初始化
