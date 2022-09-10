import sys
from PyQt5.QtWidgets import QApplication
from controller import VillageClassification
from PyQt5.QtGui import QIcon, QPixmap

def run_village():
    app = QApplication(sys.argv)
    village = VillageClassification()
    village.setWindowTitle('村落分类')
    icon = QIcon()
    icon.addPixmap(QPixmap('interface/village/resource/glass.png'))
    village.setWindowIcon(icon)
    village.show()
    sys.exit(app.exec_())

if __name__ == "__main__":
    run_village()
