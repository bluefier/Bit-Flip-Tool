import copy
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
import tensorflow.compat.v1 as tf
from keras import backend as K

from entity.info import infoWindow
import inject

class Fix():
    def __init__(self,info):
        self.ui = QUiLoader().load('ui/second.ui')  # type: ignore
        self.info = info
        self.dataset = []
        self.input_size_list = []  # type: ignore

        self.ui.pB_fix.clicked.connect(self.fix_by_layer)
        self.ui.pB_fix2.clicked.connect(self.fix_by_channel)
        self.ui.pB_upload.clicked.connect(self.upload)
        self.ui.pB_download.clicked.connect(self.download)
        # # 按下回车上传输入尺寸(不需要用户上传尺寸)
        # self.ui.lE_input_size.returnPressed.connect(self.get_input_size)
        # # 按下确定
        # self.ui.pB_sure.clicked.connect(self.get_input_size)
        # 显示框光标设置，最大输出设置
        self.ui.tB_ms.ensureCursorVisible()
        self.ui.tB_ms.document().setMaximumBlockCount(1000)
        self.info.model.summary(print_fn=self.ui.tB_ms.append)

    def upload(self):
        # print("上传数据")
        FileDirectory = QFileDialog.getExistingDirectory(QMainWindow(), "选择文件夹")
        if FileDirectory=='':
            pass
        else:
            print(FileDirectory)
            self.dataset = []
            self.ui.tB_uri.setPlainText(FileDirectory)
            # self.ui.tB_uri.append(FileDirectory) # 追加的形式

            # 读取文件夹中的图片
            files_list = os.listdir(FileDirectory)
            for filename in files_list:
                if filename.endswith(("png")):
                    absoluate_path = FileDirectory + "/" + filename
                    f = Image.open(absoluate_path) # 注意，这里不能用plt.imread()，那样会导致读不到通道，只能读取灰度图。
                    shape = [1]
                    size = []
                    for i in range(1,4):
                        shape.append(self.info.model.layers[0].input_shape[i])
                    size.append(self.info.model.layers[0].input_shape[1])
                    size.append(self.info.model.layers[0].input_shape[2])
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

    def jump(self):
        self.next = inject.Inject(self.info)
        self.next.ui.show()
        self.ui.close()
    def fix_by_layer(self):
        # 打印修复前的结构
        # for i in range(len(self.info.model.layers)):
        #     self.ui.tB_ms.append(self.info.model.layers[i])
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
            self.ui.tB_ms2.clear()
            self.info.fixed_model.summary(print_fn=self.ui.tB_ms2.append)

            # 将卷积层列表拷贝
            length = len(self.info.fixed_model.layers)
            self.info.convs = [] # 清空原模型卷积层列表，节省空间,实际上根本没有用到这个变量。
            for i in range(length):
                if "conv2d" in self.info.fixed_model.layers[i].name and "input" not in self.info.fixed_model.layers[i].name:
                    self.info.fixed_convs.append(self.info.fixed_model.layers[i])
            self.info.layer_num = len(self.info.fixed_convs)

    def fix_by_channel(self):
        # 打印修复前的结构
        # for i in range(len(self.info.model.layers)):
        #     self.ui.tB_ms.append(self.info.model.layers[i])

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
                self.ui.tB_ms2.clear()
                self.info.fixed_model.summary(print_fn=self.ui.tB_ms2.append)

                # 将卷积层列表拷贝
                length = len(self.info.fixed_model.layers)
                self.info.convs = [] # 清空原模型卷积层列表，节省空间,实际上根本没有用到这个变量。
                for i in range(length):
                    if "conv2d" in self.info.fixed_model.layers[i].name and "input" not in self.info.fixed_model.layers[i].name:
                        self.info.fixed_convs.append(self.info.fixed_model.layers[i])
                self.info.layer_num = len(self.info.fixed_convs)

    def download(self):
        if self.info.fixed_model==None:
            self.success= infoWindow("请先修复")
            self.success.ui.show()
        else:
            path=QFileDialog.getExistingDirectory(QMainWindow(), "选择文件夹")
            if path!="":
                self.info.fixed_model.save(path+"/fixed_models.h5")
                self.success = infoWindow("下载完成")
                self.success.ui.show()
                self.jump()
            else:
                self.success= infoWindow("请选择下载路径")
                self.success.ui.show()