from PySide2.QtWidgets import QApplication, QMainWindow, QFileDialog,QDialog
from PySide2.QtUiTools import QUiLoader
from PySide2.QtGui import QIcon
from PySide2.QtCore import *
from PySide2 import QtCore, QtGui, QtWidgets
# download
def upload_predict_img(self):
    predict_img_path = QFileDialog.getExistingDirectory(QMainWindow(), "选择文件夹")
    if predict_img_path=='':
        pass
    else:
        self.ui.iE_predict_img_uri.setPlainText(predict_img_path)

        f = Image.open(predict_img_path) # 注意，这里不能用plt.imread()，那样会导致读不到通道，只能读取灰度图。
        shape = [1]
        size = []
        for i in range(1,4):
            shape.append(self.info.model.layers[0].input_shape[0][i])
        size.append(self.info.model.layers[0].input_shape[0][1])
        size.append(self.info.model.layers[0].input_shape[0][2])
        # 拉伸
        f = f.resize(size, Image.ANTIALIAS)
        # 修改通道
        if shape[3]==1:
            # 说明是单通道
            f = f.convert('L')
        elif shape[3]==3:
            f = f.convert('RGB')
        else:
            print("模型输入通道数有问题")
        # 转化为np并reshape
        f = np.array(f)
        f = np.reshape(f,shape)  # type: ignore
        self.predict_img = f            

def predict(self):
    if self.predict_img!="":
        origin_result = self.info.origin_inject_model.predict(self.predict_img)
        fixed_result = self.info.fixed_inject_model.predict(self.predict_img)
        self.ui.iE_origin_result.setPlainText(origin_result)
        self.ui.iE_fixed_result.setPlainText(fixed_result)
    else:
        self.success= infoWindow("请先上传图片")
        self.success.ui.show()

def download_origin_inject(self):
    path=QFileDialog.getExistingDirectory(QMainWindow(), "选择文件夹")
    if path!="":
        self.info.origin_inject_model.save(path+"/origin_inject_model.h5")
        self.success= infoWindow("下载完成")
        self.success.ui.show()
    else:
        self.success= infoWindow("请选择下载路径")
        self.success.ui.show()
def dowload_fixed_inject(self):
    path=QFileDialog.getExistingDirectory(QMainWindow(), "选择文件夹")
    if path!="":
        self.info.fixed_inject_model.save(path+"/fixed_injected_model.h5")
        self.success= infoWindow("下载完成")
        self.success.ui.show()
    else:
        self.success= infoWindow("请选择下载路径")
        self.success.ui.show()
