from tkinter.ttk import Widget
from PySide2.QtWidgets import QApplication, QMainWindow, QFileDialog,QDialog
from PySide2.QtUiTools import QUiLoader
from PySide2.QtGui import QIcon
from PySide2.QtCore import *
from PySide2 import QtCore, QtGui, QtWidgets

import sys
import os
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from keras.models import load_model

from my_final_ui import *
from entity.info import Info, infoWindow
import fix
import inject
import utils.keras.change_relu as cr
import utils.keras.find_gate as fg

BACKGROUND_COLOR = "#222222"

TITLE_COLOR = '#D9821E' # 橙色

# 黄色：'#FFFF00'

# 按钮高度
BUTTON_HEIGHT = 18
# 按钮宽度
BUTTON_WIDTH = 39

class mwindow(QMainWindow,Ui_MainWindow):
    # 打包命令：pyinstaller main.py --noconsole --hidden-import PySide2.QtXml
    # 打包之后，需要手动将ui文件，放在dist文件夹中，路s径和load中的路径要一致（相对于exe文件）。（图片等静态资源文件同理）

    def __init__(self):
        super(mwindow, self).__init__()
        self.setupUi(self)
        self.pushButton.clicked.connect(self.close)
        self.pushButton_2.clicked.connect(self.ButtonMinSlot)

        self.setWindowFlag(QtCore.Qt.FramelessWindowHint) #去边框
        self.setAttribute(QtCore.Qt.WA_TranslucentBackground)  # 设置窗口背景透明
        self.centralwidget.setAttribute(QtCore.Qt.WA_TranslucentBackground)  # 设置pyqt自动生成的centralwidget背景透明
        self.centralwidget.setAutoFillBackground(True)
        self.pushButton.setFixedSize(QSize(BUTTON_WIDTH, BUTTON_HEIGHT))# 设置按钮大小
        self.pushButton_2.setFixedSize(QSize(BUTTON_WIDTH, BUTTON_HEIGHT)) # 设置按钮大小

        Qss =  'QWidget#widget_2{background-color: %s;}' % BACKGROUND_COLOR
        Qss += 'QWidget#widget{background-color: %s;border-top-right-radius:5 ;border-top-left-radius:5 ;}' % TITLE_COLOR
        Qss += 'QWidget#widget_3{background-color: %s;}' % TITLE_COLOR
        Qss += 'QPushButton#pushButton{background-color: %s;border-image:url(./img/btn_close_normal.png);border-top-right-radius:5 ;}' % TITLE_COLOR
        Qss += 'QPushButton#pushButton:hover{border-image:url(./img/btn_close_down2.png); border-top-right-radius:5 ;}'
        Qss += 'QPushButton#pushButton:pressed{border-image:url(./img/btn_close_down.png);border-top-right-radius:5 ;}'
        Qss += 'QPushButton#pushButton_2{background-color: %s;border-image:url(./img/btn_min_normal.png);}' % TITLE_COLOR
        Qss += 'QPushButton#pushButton_2:hover{background-color: %s;border-image:url(./img/btn_min_normal.png);}' % BACKGROUND_COLOR
        Qss += 'QPushButton#pushButton_2:pressed{background-color: %s;border-top-left-radius:5 ;}' % BACKGROUND_COLOR
        Qss += 'QPushButton#pushButton_3{background-color: %s;border-top-left-radius:5 ;border:0;}' % TITLE_COLOR
        Qss += '#label{background-color:rbga(0,0,0,0);color: %s;}' % BACKGROUND_COLOR
        self.setStyleSheet(Qss) # 边框部分qss重载

        # self.shadow=QGraphicsDropShadowEffect()
        # self.shadow.setBlurRadius(15)
        # self.shadow.setColor(QColor(0, 0, 0, 255))
        # self.shadow.setOffset(0,0)
        # self.shadow1=QGraphicsDropShadowEffect()
        # self.shadow1.setBlurRadius(15)
        # self.shadow1.setOffset(0,0)
        # self.shadow1.setColor(QColor(0, 0, 0, 255))
        # self.shadow2=QGraphicsDropShadowEffect()
        # self.shadow2.setBlurRadius(15)
        # self.shadow2.setOffset(0,0)
        # self.shadow2.setColor(QColor(0, 0, 0, 255))

        # self.widget_3.setGraphicsEffect(self.shadow2)
        # self.widget_2.setGraphicsEffect(self.shadow)
        # self.widget.setGraphicsEffect(self.shadow1) #加阴影，更立体

    def mousePressEvent(self, event):
        if event.button() == QtCore.Qt.LeftButton:
            self.m_flag = True
            self.m_Position = event.globalPos()-self.pos() #获取鼠标相对窗口的位置
            event.accept()
            self.setCursor(QCursor(QtCore.Qt.OpenHandCursor))
            
    def mouseReleaseEvent(self, event):
        if event.button() == QtCore.Qt.LeftButton:
            self.m_flag = False
            self.setCursor(QCursor(QtCore.Qt.ArrowCursor))

    def mouseMoveEvent(self, event):
        if QtCore.Qt.LeftButton and self.m_flag:
            self.move(event.globalPos() - self.m_Position) #更改窗口位置
            event.accept()

    def ButtonMinSlot(self):
        self.showMinimized()

if __name__ == '__main__':
    app = QApplication(sys.argv)
    w = mwindow()
    with open('qss/DarkOrangeQss.txt') as file:
        str = file.readlines()
        str = ''.join(str).strip('\n')
        app.setStyleSheet(str)
    w.show()
    sys.exit(app.exec_())