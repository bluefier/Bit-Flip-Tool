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

from entity.info import Info, infoWindow
import fix
import inject

class Main():
    # 打包命令：pyinstaller main.py --noconsole --hidden-import PySide2.QtXml
    # 打包之后，需要手动将ui文件，放在dist文件夹中，路s径和load中的路径要一致（相对于exe文件）。（图片等静态资源文件同理）
    def __init__(self):
        self.ui = QUiLoader().load('ui/main.ui')  # type: ignore
        self.info = Info()
        self.ui.pB_sure.clicked.connect(self.jump)
        # 网络和数据集修改
        self.ui.pB_upload.clicked.connect(self.upload)
        # self.ui.pB_upload_structure.clicked.connect(self.upload_structure)
    def upload(self):
        file = QFileDialog.getOpenFileName(QMainWindow(), "选择文件",filter="model (*.h5)")
        if file[0]=='':
            pass
        else:
            # print(file[0])
            self.ui.tB_uri.setPlainText(file[0])
            
            # self.ui.tB_uri.append(file[0]) # 追加的形式
            self.info.model = load_model(file[0])
            length = len(self.info.model.layers)
            for i in range(length):
                if "conv2d" in self.info.model.layers[i].name and "input" not in self.info.model.layers[i].name:   
                    self.info.convs.append(self.info.model.layers[i])
            self.info.layer_num = len(self.info.convs)

    # def upload_structure(self):
    #     file = QFileDialog.getOpenFileName(QMainWindow(), "选择文件",filter="net (*.py)")
    #     if file[0]=='':
    #         pass
    #     else:
    #         print(file[0])
    #         self.ui.tB_uri_structure.setPlainText(file[0])
    #         # self.ui.tB_uri.append(file[0]) # 追加的形式
    #         # self.info.net = file[0]

    def jump(self):
        if self.info.model==None:
            self.success= infoWindow("请选择正确模型")
            self.success.ui.show()
        else:
            self.next = fix.Fix(self.info)
            self.next.ui.show()
            self.ui.close()   

        # self.next = inject.Inject(self.info)
        # self.next.ui.show()
        # self.ui.close()

        
if __name__ == '__main__':
    app = QApplication([])
    # app.setWindowIcon(QIcon('logo.png'))

    m = Main()
    m.ui.show()
    
    app.exec_()