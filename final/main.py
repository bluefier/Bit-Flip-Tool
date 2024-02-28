from PySide2.QtWidgets import QApplication, QMainWindow, QFileDialog,QDialog
from PySide2.QtUiTools import QUiLoader
from PySide2.QtGui import QIcon
from PySide2.QtCore import *
from PySide2 import QtCore, QtGui, QtWidgets


# show
def update_models_print(self,x):
    if x ==1:
        self.ui.tB_origin_model.clear()
        self.info.model.summary(print_fn=self.ui.tB_origin_model.append)
    if x ==2:
        self.ui.tB_fixed_model.clear()
        self.info.fixed_model.summary(print_fn=self.ui.tB_fixed_model.append)
    if x ==3:
        self.ui.tB_origin_model.clear()
        self.info.origin_inject_model.summary(print_fn=self.ui.tB_origin_model.append)
    if x == 4:
        self.ui.tB_fixed_model.clear()
        self.info.fixed_inject_model.summary(print_fn=self.ui.tB_fixed_model.append)

# main
def upload_model(self):
    file = QFileDialog.getOpenFileName(QMainWindow(), "选择文件",filter="model (*.h5)")
    if file[0]=='':
        pass
    else:
        # print(file[0])
        self.ui.iE_model_uri.setPlainText(file[0])
        
        # self.ui.tB_uri.append(file[0]) # 追加的形式
        self.info.model = load_model(file[0])
        length = len(self.info.model.layers)
        for i in range(length):
            if "conv2d" in self.info.model.layers[i].name and "input" not in self.info.model.layers[i].name:   
                self.info.convs.append(self.info.model.layers[i])
        self.info.layer_num = len(self.info.convs)
        self.update_models_print(1)

def upload_data(self):
    # print("上传数据")
    self.data_dir = QFileDialog.getExistingDirectory(QMainWindow(), "选择文件夹")
    if self.data_dir=='':
        pass
    else:
        # print(FileDirectory)
        self.dataset = []
        self.ui.iE_data_uri.setPlainText(self.data_dir)
        # self.ui.tB_uri.append(FileDirectory) # 追加的形式
        

