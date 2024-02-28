import os

from util import *
from PySide2.QtWidgets import QApplication, QMainWindow, QFileDialog,QDialog
from PySide2.QtUiTools import QUiLoader
from PySide2.QtGui import QIcon
from PySide2.QtCore import *
from PySide2 import QtCore, QtGui, QtWidgets
# fix
def load_data(self,x):
    # 读取文件夹中的图片
    files_list = os.listdir(self.data_dir)
    for filename in files_list:
        if filename.endswith(("png")):
            absoluate_path = self.data_dir + "/" + filename
            f = Image.open(absoluate_path) # 注意，这里不能用plt.imread()，那样会导致读不到通道，只能读取灰度图。
            shape = [1]
            size = []
            for i in range(1,4):
                shape.append(self.info.model.layers[0].input_shape[0][i])
            size.append(self.info.model.layers[0].input_shape[0][1])
            size.append(self.info.model.layers[0].input_shape[0][2])
            # for layer in self.info.model.layers:
            #     print(layer.input_shape)

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
            # print("resize后的形状:"+str(f.shape))
            f = np.reshape(f,shape)  # type: ignore
            # print(f.shape)
            self.dataset.append(f)
            # print("数据集长度：{}".format(len(self.dataset)))
    if self.dataset==[]:
        self.infoWindow = infoWindow("图片需要为png格式")
        self.infoWindow.ui.show()
def fix(self):
    method = self.ui.cB_fix_method.value()
    if method == "BRelu":
        self.fix_by_layer()
    if method =="ChRelu":
        self.fix_by_channel()
    if method =="FBRelu":
        self.fix_by_FBRelu()

def fix_by_layer(self):
    if self.dataset ==[]:
        self.infoWindow = infoWindow("请先上传图片")
        self.infoWindow.ui.show()
    else:
        # 计算上边界
        boundarys = fg.calculate_boundary(self.dataset,self.info.model)  # type: ignore
        # 根据上边界来fix
        # print(boundarys)
        self.info.fixed_model = cr.fix(self.info.model,boundarys)
        # 将原模型权重给新模型。
        weights = self.info.model.get_weights()
        self.info.fixed_model.set_weights(weights)
        # self.info.fixed_model.compile(optimizer="rmsprop", loss='categorical_crossentropy', metrics=["accuracy"])

        # print(self.info.fixed_model.__dict__)
        # print(self.info.model.__dict__)
        # self.info.fixed_model.compile = self.info.model.compile

        # 弹窗打印修复成功
        self.success= infoWindow("修复成功")
        self.success.ui.show()
        # 显示修复成功后的结构
        self.update_models_print(2)

        # 将卷积层列表拷贝（useless）
        length = len(self.info.fixed_model.layers)
        self.info.convs = [] # 清空原模型卷积层列表，节省空间,实际上根本没有用到这个变量。
        for i in range(length):
            if "conv2d" in self.info.fixed_model.layers[i].name and "input" not in self.info.fixed_model.layers[i].name:
                self.info.fixed_convs.append(self.info.fixed_model.layers[i])
        self.info.layer_num = len(self.info.fixed_convs)

def fix_by_channel(self):
    if self.dataset ==[]:
        self.infoWindow = infoWindow("请先上传图片")
        self.infoWindow.ui.show()
    else:
        # 计算上边界
        boundarys = fg.calculate_ch_boundary(self.dataset,self.info.model)  # type: ignore
        # 根据上边界来fix
        # print(boundarys)
        if boundarys == False:
            self.success= infoWindow("某层通道数超过500")
            self.success.ui.show()
        else:
            self.info.fixed_model = cr.fix_ch(self.info.model,boundarys)
            weights = self.info.model.get_weights()
            self.info.fixed_model.set_weights(weights)

            # 弹窗打印修复成功
            self.success= infoWindow("修复成功")
            self.success.ui.show()
            # 显示修复成功后的结构
            self.update_models_print(2)

            # 将卷积层列表拷贝(useless)
            length = len(self.info.fixed_model.layers)
            self.info.convs = [] # 清空原模型卷积层列表，节省空间,实际上根本没有用到这个变量。
            for i in range(length):
                if "conv2d" in self.info.fixed_model.layers[i].name and "input" not in self.info.fixed_model.layers[i].name:
                    self.info.fixed_convs.append(self.info.fixed_model.layers[i])
            self.info.layer_num = len(self.info.fixed_convs)

def download_fixed(self):
    if self.info.fixed_model==None:
        self.success= infoWindow("请先修复")
        self.success.ui.show()
    else:
        path=QFileDialog.getExistingDirectory(QMainWindow(), "选择文件夹")
        if path!="":
            self.info.fixed_model.save(path+"/fixed_models.h5")
            self.success = infoWindow("下载完成")
            self.success.ui.show()
        else:
            self.success= infoWindow("请选择下载路径")
            self.success.ui.show()
