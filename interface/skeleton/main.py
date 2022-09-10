import sys
from PyQt5.QtWidgets import QApplication
from interface.skeleton.controller import Skeleton
from PyQt5.QtGui import QIcon, QPixmap


def run_skeleton():
    app = QApplication(sys.argv)
    skeleton = Skeleton()
    skeleton.setWindowTitle('村落骨架提取')
    icon = QIcon()
    icon.addPixmap(QPixmap('interface/skeleton/resource/glass.png'))
    skeleton.setWindowIcon(icon)
    skeleton.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    run_skeleton()
