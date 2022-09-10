import sys
from PyQt5.QtWidgets import QApplication
from controller import RoadExtract
from PyQt5.QtGui import QIcon, QPixmap

def run_road():
    app = QApplication(sys.argv)
    roadExtract = RoadExtract()
    roadExtract.setWindowTitle('道路提取')
    # 设置程序图标
    icon = QIcon()
    icon.addPixmap(QPixmap('interface/mountain/resource/glass.png'))
    roadExtract.setWindowIcon(icon)
    # 主界面显示
    roadExtract.show()
    sys.exit(app.exec_())

if __name__ == "__main__":
    run_road()