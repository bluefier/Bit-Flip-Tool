from tkinter.ttk import Widget
from PySide2.QtWidgets import QApplication, QMainWindow, QFileDialog,QDialog
from PySide2.QtUiTools import QUiLoader
from PySide2.QtGui import QIcon
from keras.models import load_model
import utils.keras.change_relu as cr
import utils.keras.find_gate as fg
import os
# import cv2
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

from entity.info import Info,infoWindow
import inject
import main

class Download:
    def __init__(self,info):
        self.ui = QUiLoader().load('ui/fourth.ui')  # type: ignore
        self.info = info

        self.ui.pB_back.clicked.connect(self.go_to_main)
        self.ui.pB_refine.clicked.connect(self.go_to_inject)
        self.ui.pB_download.clicked.connect(self.download_fixed_injected_model)
        self.ui.pB_download_2.clicked.connect(self.download_normal_inject_model)

    def download_normal_inject_model(self):
        # print("download")
        path=QFileDialog.getExistingDirectory(QMainWindow(), "选择文件夹")
        if path!="":
            self.info.normal_inject_model.save(path+"/normal_inject_model.h5")
            self.success= infoWindow("下载完成")
            self.success.ui.show()
        else:
            self.success= infoWindow("请选择下载路径")
            self.success.ui.show()
    def download_fixed_injected_model(self):
        # print("download")
        path=QFileDialog.getExistingDirectory(QMainWindow(), "选择文件夹")
        if path!="":
            self.info.fixed_inject_model.save(path+"/fixed_injected_model.h5")
            self.success= infoWindow("下载完成")
            self.success.ui.show()
        else:
            self.success= infoWindow("请选择下载路径")
            self.success.ui.show()
            
    def go_to_main(self):
        self.next = main.Main()
        self.next.ui.show()
        self.ui.close()
    def go_to_inject(self):
        # 需要将错误注入的模型还原
        self.next = inject.Inject(self.info)
        self.next.ui.show()
        self.ui.close()