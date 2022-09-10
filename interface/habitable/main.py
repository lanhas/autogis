import sys
from PyQt5.QtWidgets import QApplication
from controller import HabitablePred
from PyQt5.QtGui import QIcon, QPixmap

def run_habitable():
    app = QApplication(sys.argv)
    habitablePred = HabitablePred()
    habitablePred.setWindowTitle('宜居区域预测')
    # 设置程序图标
    icon = QIcon()
    icon.addPixmap(QPixmap('interface/mountain/resource/glass.png'))
    habitablePred.setWindowIcon(icon)
    # 主界面显示
    habitablePred.show()
    sys.exit(app.exec_())

if __name__ == "__main__":
    run_habitable()