# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'f:\code\python\SemanticSegmentation\autogis\interface\axis_trans\parameters.ui'
#
# Created by: PyQt5 UI code generator 5.15.4
#
# WARNING: Any manual changes made to this file will be lost when pyuic5 is
# run again.  Do not edit this file unless you know what you are doing.


from PyQt5 import QtCore, QtGui, QtWidgets


class Ui_Form(object):
    def setupUi(self, Form):
        Form.setObjectName("Form")
        Form.resize(397, 377)
        self.gridLayout = QtWidgets.QGridLayout(Form)
        self.gridLayout.setObjectName("gridLayout")
        self.widget = QtWidgets.QWidget(Form)
        self.widget.setObjectName("widget")
        self.horizontalLayout = QtWidgets.QHBoxLayout(self.widget)
        self.horizontalLayout.setObjectName("horizontalLayout")
        self.label = QtWidgets.QLabel(self.widget)
        self.label.setObjectName("label")
        self.horizontalLayout.addWidget(self.label)
        self.lineEdit_kernelSize = QtWidgets.QLineEdit(self.widget)
        self.lineEdit_kernelSize.setObjectName("lineEdit_kernelSize")
        self.horizontalLayout.addWidget(self.lineEdit_kernelSize)
        self.gridLayout.addWidget(self.widget, 0, 0, 1, 1)
        self.widget_2 = QtWidgets.QWidget(Form)
        self.widget_2.setObjectName("widget_2")
        self.horizontalLayout_2 = QtWidgets.QHBoxLayout(self.widget_2)
        self.horizontalLayout_2.setObjectName("horizontalLayout_2")
        self.label_2 = QtWidgets.QLabel(self.widget_2)
        self.label_2.setObjectName("label_2")
        self.horizontalLayout_2.addWidget(self.label_2)
        self.lineEdit_sleepTime = QtWidgets.QLineEdit(self.widget_2)
        self.lineEdit_sleepTime.setObjectName("lineEdit_sleepTime")
        self.horizontalLayout_2.addWidget(self.lineEdit_sleepTime)
        self.gridLayout.addWidget(self.widget_2, 1, 0, 1, 1)
        self.widget_3 = QtWidgets.QWidget(Form)
        self.widget_3.setObjectName("widget_3")
        self.horizontalLayout_3 = QtWidgets.QHBoxLayout(self.widget_3)
        self.horizontalLayout_3.setObjectName("horizontalLayout_3")
        self.label_3 = QtWidgets.QLabel(self.widget_3)
        self.label_3.setObjectName("label_3")
        self.horizontalLayout_3.addWidget(self.label_3)
        self.lineEdit_iterNum = QtWidgets.QLineEdit(self.widget_3)
        self.lineEdit_iterNum.setObjectName("lineEdit_iterNum")
        self.horizontalLayout_3.addWidget(self.lineEdit_iterNum)
        self.gridLayout.addWidget(self.widget_3, 0, 1, 1, 1)
        self.widget_7 = QtWidgets.QWidget(Form)
        self.widget_7.setObjectName("widget_7")
        self.horizontalLayout_7 = QtWidgets.QHBoxLayout(self.widget_7)
        self.horizontalLayout_7.setObjectName("horizontalLayout_7")
        self.label_7 = QtWidgets.QLabel(self.widget_7)
        self.label_7.setObjectName("label_7")
        self.horizontalLayout_7.addWidget(self.label_7)
        self.comboBox_axisColor = QtWidgets.QComboBox(self.widget_7)
        self.comboBox_axisColor.setObjectName("comboBox_axisColor")
        self.comboBox_axisColor.addItem("")
        self.comboBox_axisColor.addItem("")
        self.comboBox_axisColor.addItem("")
        self.comboBox_axisColor.addItem("")
        self.comboBox_axisColor.addItem("")
        self.comboBox_axisColor.addItem("")
        self.comboBox_axisColor.addItem("")
        self.comboBox_axisColor.addItem("")
        self.comboBox_axisColor.addItem("")
        self.comboBox_axisColor.addItem("")
        self.horizontalLayout_7.addWidget(self.comboBox_axisColor)
        self.gridLayout.addWidget(self.widget_7, 4, 1, 1, 1)
        self.widget_9 = QtWidgets.QWidget(Form)
        self.widget_9.setObjectName("widget_9")
        self.horizontalLayout_9 = QtWidgets.QHBoxLayout(self.widget_9)
        self.horizontalLayout_9.setObjectName("horizontalLayout_9")
        self.label_9 = QtWidgets.QLabel(self.widget_9)
        self.label_9.setObjectName("label_9")
        self.horizontalLayout_9.addWidget(self.label_9)
        self.lineEdit_gradWe = QtWidgets.QLineEdit(self.widget_9)
        self.lineEdit_gradWe.setObjectName("lineEdit_gradWe")
        self.horizontalLayout_9.addWidget(self.lineEdit_gradWe)
        self.gridLayout.addWidget(self.widget_9, 2, 1, 1, 1)
        self.widget_10 = QtWidgets.QWidget(Form)
        self.widget_10.setObjectName("widget_10")
        self.horizontalLayout_10 = QtWidgets.QHBoxLayout(self.widget_10)
        self.horizontalLayout_10.setObjectName("horizontalLayout_10")
        self.label_10 = QtWidgets.QLabel(self.widget_10)
        self.label_10.setObjectName("label_10")
        self.horizontalLayout_10.addWidget(self.label_10)
        self.lineEdit_gradSn = QtWidgets.QLineEdit(self.widget_10)
        self.lineEdit_gradSn.setObjectName("lineEdit_gradSn")
        self.horizontalLayout_10.addWidget(self.lineEdit_gradSn)
        self.gridLayout.addWidget(self.widget_10, 2, 0, 1, 1)
        self.widget_11 = QtWidgets.QWidget(Form)
        self.widget_11.setObjectName("widget_11")
        self.horizontalLayout_11 = QtWidgets.QHBoxLayout(self.widget_11)
        self.horizontalLayout_11.setObjectName("horizontalLayout_11")
        self.label_11 = QtWidgets.QLabel(self.widget_11)
        self.label_11.setObjectName("label_11")
        self.horizontalLayout_11.addWidget(self.label_11)
        self.lineEdit_slopeThreshold = QtWidgets.QLineEdit(self.widget_11)
        self.lineEdit_slopeThreshold.setObjectName("lineEdit_slopeThreshold")
        self.horizontalLayout_11.addWidget(self.lineEdit_slopeThreshold)
        self.gridLayout.addWidget(self.widget_11, 1, 1, 1, 1)
        self.widget_6 = QtWidgets.QWidget(Form)
        self.widget_6.setObjectName("widget_6")
        self.horizontalLayout_6 = QtWidgets.QHBoxLayout(self.widget_6)
        self.horizontalLayout_6.setObjectName("horizontalLayout_6")
        self.label_6 = QtWidgets.QLabel(self.widget_6)
        self.label_6.setObjectName("label_6")
        self.horizontalLayout_6.addWidget(self.label_6)
        self.comboBox_contourPenCol = QtWidgets.QComboBox(self.widget_6)
        self.comboBox_contourPenCol.setObjectName("comboBox_contourPenCol")
        self.comboBox_contourPenCol.addItem("")
        self.comboBox_contourPenCol.addItem("")
        self.comboBox_contourPenCol.addItem("")
        self.comboBox_contourPenCol.addItem("")
        self.comboBox_contourPenCol.addItem("")
        self.comboBox_contourPenCol.addItem("")
        self.comboBox_contourPenCol.addItem("")
        self.comboBox_contourPenCol.addItem("")
        self.comboBox_contourPenCol.addItem("")
        self.comboBox_contourPenCol.addItem("")
        self.horizontalLayout_6.addWidget(self.comboBox_contourPenCol)
        self.gridLayout.addWidget(self.widget_6, 4, 0, 1, 1)
        self.label_5 = QtWidgets.QLabel(Form)
        font = QtGui.QFont()
        font.setPointSize(15)
        font.setBold(False)
        font.setWeight(50)
        self.label_5.setFont(font)
        self.label_5.setAlignment(QtCore.Qt.AlignCenter)
        self.label_5.setObjectName("label_5")
        self.gridLayout.addWidget(self.label_5, 3, 0, 1, 1)
        self.widget_4 = QtWidgets.QWidget(Form)
        self.widget_4.setObjectName("widget_4")
        self.horizontalLayout_4 = QtWidgets.QHBoxLayout(self.widget_4)
        self.horizontalLayout_4.setObjectName("horizontalLayout_4")
        self.label_4 = QtWidgets.QLabel(self.widget_4)
        self.label_4.setObjectName("label_4")
        self.horizontalLayout_4.addWidget(self.label_4)
        self.comboBox_outlineColor = QtWidgets.QComboBox(self.widget_4)
        self.comboBox_outlineColor.setObjectName("comboBox_outlineColor")
        self.comboBox_outlineColor.addItem("")
        self.comboBox_outlineColor.addItem("")
        self.comboBox_outlineColor.addItem("")
        self.comboBox_outlineColor.addItem("")
        self.comboBox_outlineColor.addItem("")
        self.comboBox_outlineColor.addItem("")
        self.comboBox_outlineColor.addItem("")
        self.comboBox_outlineColor.addItem("")
        self.comboBox_outlineColor.addItem("")
        self.comboBox_outlineColor.addItem("")
        self.horizontalLayout_4.addWidget(self.comboBox_outlineColor)
        self.gridLayout.addWidget(self.widget_4, 3, 1, 1, 1)
        self.widget_12 = QtWidgets.QWidget(Form)
        self.widget_12.setObjectName("widget_12")
        self.horizontalLayout_12 = QtWidgets.QHBoxLayout(self.widget_12)
        self.horizontalLayout_12.setObjectName("horizontalLayout_12")
        self.pushButton_default = QtWidgets.QPushButton(self.widget_12)
        self.pushButton_default.setObjectName("pushButton_default")
        self.horizontalLayout_12.addWidget(self.pushButton_default)
        self.pushButton_submit = QtWidgets.QPushButton(self.widget_12)
        self.pushButton_submit.setObjectName("pushButton_submit")
        self.horizontalLayout_12.addWidget(self.pushButton_submit)
        self.gridLayout.addWidget(self.widget_12, 6, 1, 1, 1)
        self.widget_8 = QtWidgets.QWidget(Form)
        self.widget_8.setObjectName("widget_8")
        self.horizontalLayout_5 = QtWidgets.QHBoxLayout(self.widget_8)
        self.horizontalLayout_5.setObjectName("horizontalLayout_5")
        self.label_8 = QtWidgets.QLabel(self.widget_8)
        self.label_8.setObjectName("label_8")
        self.horizontalLayout_5.addWidget(self.label_8)
        self.comboBox_roadPenCol = QtWidgets.QComboBox(self.widget_8)
        self.comboBox_roadPenCol.setObjectName("comboBox_roadPenCol")
        self.comboBox_roadPenCol.addItem("")
        self.comboBox_roadPenCol.addItem("")
        self.comboBox_roadPenCol.addItem("")
        self.comboBox_roadPenCol.addItem("")
        self.comboBox_roadPenCol.addItem("")
        self.comboBox_roadPenCol.addItem("")
        self.comboBox_roadPenCol.addItem("")
        self.comboBox_roadPenCol.addItem("")
        self.comboBox_roadPenCol.addItem("")
        self.comboBox_roadPenCol.addItem("")
        self.horizontalLayout_5.addWidget(self.comboBox_roadPenCol)
        self.gridLayout.addWidget(self.widget_8, 5, 0, 1, 1)
        self.widget_5 = QtWidgets.QWidget(Form)
        self.widget_5.setObjectName("widget_5")
        self.horizontalLayout_8 = QtWidgets.QHBoxLayout(self.widget_5)
        self.horizontalLayout_8.setObjectName("horizontalLayout_8")
        self.label_12 = QtWidgets.QLabel(self.widget_5)
        self.label_12.setObjectName("label_12")
        self.horizontalLayout_8.addWidget(self.label_12)
        self.comboBox_axisWidth = QtWidgets.QComboBox(self.widget_5)
        font = QtGui.QFont()
        font.setBold(False)
        font.setWeight(50)
        self.comboBox_axisWidth.setFont(font)
        self.comboBox_axisWidth.setObjectName("comboBox_axisWidth")
        self.comboBox_axisWidth.addItem("")
        self.comboBox_axisWidth.addItem("")
        self.comboBox_axisWidth.addItem("")
        self.comboBox_axisWidth.addItem("")
        self.comboBox_axisWidth.addItem("")
        self.horizontalLayout_8.addWidget(self.comboBox_axisWidth)
        self.gridLayout.addWidget(self.widget_5, 5, 1, 1, 1)

        self.retranslateUi(Form)
        self.comboBox_axisColor.setCurrentIndex(1)
        self.comboBox_roadPenCol.setCurrentIndex(5)
        self.comboBox_axisWidth.setCurrentIndex(1)
        self.pushButton_default.clicked.connect(Form.reset)
        self.pushButton_submit.clicked.connect(Form.commit)
        QtCore.QMetaObject.connectSlotsByName(Form)

    def retranslateUi(self, Form):
        _translate = QtCore.QCoreApplication.translate
        Form.setWindowTitle(_translate("Form", "Form"))
        self.label.setText(_translate("Form", "卷积核大小："))
        self.lineEdit_kernelSize.setText(_translate("Form", "8"))
        self.label_2.setText(_translate("Form", "动态显示间隔时间："))
        self.lineEdit_sleepTime.setText(_translate("Form", "0.02"))
        self.label_3.setText(_translate("Form", "迭代次数："))
        self.lineEdit_iterNum.setText(_translate("Form", "13"))
        self.label_7.setText(_translate("Form", "轴线颜色："))
        self.comboBox_axisColor.setItemText(0, _translate("Form", "红色"))
        self.comboBox_axisColor.setItemText(1, _translate("Form", "橙色"))
        self.comboBox_axisColor.setItemText(2, _translate("Form", "黄色"))
        self.comboBox_axisColor.setItemText(3, _translate("Form", "绿色"))
        self.comboBox_axisColor.setItemText(4, _translate("Form", "青色"))
        self.comboBox_axisColor.setItemText(5, _translate("Form", "蓝色"))
        self.comboBox_axisColor.setItemText(6, _translate("Form", "紫色"))
        self.comboBox_axisColor.setItemText(7, _translate("Form", "黑色"))
        self.comboBox_axisColor.setItemText(8, _translate("Form", "白色"))
        self.comboBox_axisColor.setItemText(9, _translate("Form", "灰色"))
        self.label_9.setText(_translate("Form", "格网宽度："))
        self.lineEdit_gradWe.setText(_translate("Form", "0.53"))
        self.label_10.setText(_translate("Form", "格网高度："))
        self.lineEdit_gradSn.setText(_translate("Form", "0.53"))
        self.label_11.setText(_translate("Form", "坡度阈值："))
        self.lineEdit_slopeThreshold.setText(_translate("Form", "15"))
        self.label_6.setText(_translate("Form", "轮廓线颜色："))
        self.comboBox_contourPenCol.setItemText(0, _translate("Form", "红色"))
        self.comboBox_contourPenCol.setItemText(1, _translate("Form", "橙色"))
        self.comboBox_contourPenCol.setItemText(2, _translate("Form", "黄色"))
        self.comboBox_contourPenCol.setItemText(3, _translate("Form", "绿色"))
        self.comboBox_contourPenCol.setItemText(4, _translate("Form", "青色"))
        self.comboBox_contourPenCol.setItemText(5, _translate("Form", "蓝色"))
        self.comboBox_contourPenCol.setItemText(6, _translate("Form", "紫色"))
        self.comboBox_contourPenCol.setItemText(7, _translate("Form", "黑色"))
        self.comboBox_contourPenCol.setItemText(8, _translate("Form", "白色"))
        self.comboBox_contourPenCol.setItemText(9, _translate("Form", "灰色"))
        self.label_5.setText(_translate("Form", "显示设置"))
        self.label_4.setText(_translate("Form", "边界线颜色："))
        self.comboBox_outlineColor.setItemText(0, _translate("Form", "红色"))
        self.comboBox_outlineColor.setItemText(1, _translate("Form", "橙色"))
        self.comboBox_outlineColor.setItemText(2, _translate("Form", "黄色"))
        self.comboBox_outlineColor.setItemText(3, _translate("Form", "绿色"))
        self.comboBox_outlineColor.setItemText(4, _translate("Form", "青色"))
        self.comboBox_outlineColor.setItemText(5, _translate("Form", "蓝色"))
        self.comboBox_outlineColor.setItemText(6, _translate("Form", "紫色"))
        self.comboBox_outlineColor.setItemText(7, _translate("Form", "黑色"))
        self.comboBox_outlineColor.setItemText(8, _translate("Form", "白色"))
        self.comboBox_outlineColor.setItemText(9, _translate("Form", "灰色"))
        self.pushButton_default.setText(_translate("Form", "恢复默认"))
        self.pushButton_submit.setText(_translate("Form", "提交"))
        self.label_8.setText(_translate("Form", "道路线颜色："))
        self.comboBox_roadPenCol.setItemText(0, _translate("Form", "红色"))
        self.comboBox_roadPenCol.setItemText(1, _translate("Form", "橙色"))
        self.comboBox_roadPenCol.setItemText(2, _translate("Form", "黄色"))
        self.comboBox_roadPenCol.setItemText(3, _translate("Form", "绿色"))
        self.comboBox_roadPenCol.setItemText(4, _translate("Form", "青色"))
        self.comboBox_roadPenCol.setItemText(5, _translate("Form", "蓝色"))
        self.comboBox_roadPenCol.setItemText(6, _translate("Form", "紫色"))
        self.comboBox_roadPenCol.setItemText(7, _translate("Form", "黑色"))
        self.comboBox_roadPenCol.setItemText(8, _translate("Form", "白色"))
        self.comboBox_roadPenCol.setItemText(9, _translate("Form", "灰色"))
        self.label_12.setText(_translate("Form", "轴线宽度："))
        self.comboBox_axisWidth.setItemText(0, _translate("Form", "1"))
        self.comboBox_axisWidth.setItemText(1, _translate("Form", "2"))
        self.comboBox_axisWidth.setItemText(2, _translate("Form", "3"))
        self.comboBox_axisWidth.setItemText(3, _translate("Form", "4"))
        self.comboBox_axisWidth.setItemText(4, _translate("Form", "5"))