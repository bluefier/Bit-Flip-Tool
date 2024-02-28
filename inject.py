import copy
from tkinter.ttk import Widget
from PySide2.QtWidgets import QApplication, QMainWindow, QFileDialog,QDialog
from PySide2.QtUiTools import QUiLoader
from PySide2.QtGui import QIcon
from keras.models import load_model
import os
# import cv2
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from keras.models import clone_model

from entity.info import Info,infoWindow
from utils.keras.fault_inject import *
import download

class Inject():
    def __init__(self,info):
        self.ui = QUiLoader().load('ui/third.ui')  # type: ignore
        self.info = info
        # 拷贝模型（保证每次回来都是在修复模型上重新错误注入）
        self.info.fixed_inject_model = clone_model(self.info.fixed_model)
        weights = self.info.fixed_model.get_weights()
        self.info.fixed_inject_model.set_weights(weights)
        # 存储所有卷积层
        self.fixed_inject_convs = []
        length = len(self.info.fixed_inject_model.layers)
        for i in range(length):
            if "conv2d" in self.info.fixed_inject_model.layers[i].name and "input" not in self.info.fixed_inject_model.layers[i].name:
                self.fixed_inject_convs.append(self.info.fixed_inject_model.layers[i])

        self.info.normal_inject_model = clone_model(self.info.model)
        weights = self.info.model.get_weights()
        self.info.normal_inject_model.set_weights(weights)

        self.normal_inject_convs = []
        length = len(self.info.normal_inject_model.layers)
        for i in range(length):
            if "conv2d" in self.info.normal_inject_model.layers[i].name and "input" not in self.info.normal_inject_model.layers[i].name:
                self.normal_inject_convs.append(self.info.normal_inject_model.layers[i])

        self.ui.pB_inject.clicked.connect(self.fixed_inject_check)
        self.ui.pB_inject_2.clicked.connect(self.normal_inject_check)
        self.ui.pB_next.clicked.connect(self.jump)
        self.ui.tabWidget.currentChanged.connect(self.tab_change)
        self.ui.sB_layer.valueChanged.connect(self.layer_change)
        self.ui.sB_layer2.valueChanged.connect(self.layer_change2) # 必须分开，不能在一个函数中用if页面来判断。会有逻辑错误。

        # 参数
        self.cur_page = 0
        self.mintarget = -100000
        self.maxtarget = 100000
        # 方式一
        self.layer = 0  # 从第一层开始。
        self.location = [0,0,0,0]
        self.exp_location = 31
        # 方式二
        self.rate = 10
        # 方式三
        self.layer2 = 0
        self.location2 = [0,0,0,0]
        self.target = 99

        # 设置值(# ui里需要将所有值设置为0，后面修改，就必先走change函数，就必有self.weight和self.fixed_Layer)
        self.ui.sB_layer.setValue(self.info.layer_num)
        self.ui.sB_rate.setValue(6)
        self.ui.sB_layer2.setValue(self.info.layer_num)
        # 设置最大值和最小值
        self.ui.sB_layer.setRange(1,self.info.layer_num)
        self.ui.sB_exp_location.setRange(1,32)
        self.ui.sB_rate.setRange(1,60)
        self.ui.sB_layer2.setRange(1,self.info.layer_num)
        self.ui.sB_target.setRange(self.mintarget,self.maxtarget)

    def layer_change(self):
        print("单比特层修改")
        self.layer = self.ui.sB_layer.value()
        # print(self.layer)
        self.fixed_Layer = self.fixed_inject_convs[self.layer-1] # 因为convs下标从0开始，但用户输出最小值为1，所以拿的时候要-1
        self.weight,self.bias = self.fixed_Layer.get_weights() # (weight,bias) 才叫weight
        compares = self.weight.shape # 两个shape都是一样的

        self.normal_Layer = self.normal_inject_convs[self.layer-1]
        self.normal_weight,self.normal_bias = self.normal_Layer.get_weights()

        # 设置范围(必须先设置范围)
        self.ui.sB_location1.setRange(1,compares[0])
        self.ui.sB_location2.setRange(1,compares[1])
        self.ui.sB_location3.setRange(1,compares[2])
        self.ui.sB_location4.setRange(1,compares[3])

        self.ui.sB_location1.setValue(compares[0])
        self.ui.sB_location2.setValue(compares[1])
        self.ui.sB_location3.setValue(compares[2])
        self.ui.sB_location4.setValue(compares[3])

    def layer_change2(self):
        print("指定数的层修改")
        self.layer2 = self.ui.sB_layer2.value()
        # print(self.layer2)
        self.fixed_Layer2 = self.fixed_inject_convs[self.layer2-1]
        self.weight2,self.bias2 = self.fixed_Layer2.get_weights()
        compares = self.weight2.shape

        self.normal_Layer2 = self.normal_inject_convs[self.layer2-1]
        self.normal_weight2,self.normal_bias2 = self.normal_Layer2.get_weights()

        # 设置范围
        self.ui.sB_location2_1.setRange(1,compares[0])
        self.ui.sB_location2_2.setRange(1,compares[1])
        self.ui.sB_location2_3.setRange(1,compares[2])
        self.ui.sB_location2_4.setRange(1,compares[3])

        self.ui.sB_location2_1.setValue(compares[0])
        self.ui.sB_location2_2.setValue(compares[1])
        self.ui.sB_location2_3.setValue(compares[2])
        self.ui.sB_location2_4.setValue(compares[3])

    def tab_change(self,x):
        # 下标从0开始
        self.cur_page =x

    def jump(self):
        if self.info.normal_inject_model==None or self.info.fixed_inject_model==None:
            self.success= infoWindow("需要将两种模型都注入错误")
            self.success.ui.show()
        else:
            self.next = download.Download(self.info)
            self.next.ui.show()
            self.ui.close()

    def normal_inject_check(self):
        # 注入的时候判断当前是哪一页，并且根据判断读取对应页面参数。
        # 将页面参数存入info.fp中,因为有多个接口要用
        if self.cur_page == 0:
            self.layer = self.ui.sB_layer.value()
            self.location = [self.ui.sB_location1.value(),self.ui.sB_location2.value(),self.ui.sB_location3.value(),self.ui.sB_location4.value()]
            self.exp_location = self.ui.sB_exp_location.value()
        elif self.cur_page == 1:
            self.rate = self.ui.sB_rate.value()
        else:
            self.layer2 = self.ui.sB_layer2.value()
            self.location2 = [self.ui.sB_location2_1.value(),self.ui.sB_location2_2.value(),self.ui.sB_location2_3.value(),self.ui.sB_location2_4.value()]
            self.target = self.ui.sB_target.value()
        if self.check():
            # 注入
            self.normal_inject()
            self.success= infoWindow("注入成功")
            self.success.ui.show()
        else:
            self.success= infoWindow("请确定参数范围，输入正确参数")
            self.success.ui.show()
    def normal_inject(self):
        if self.cur_page==0:
            # 针对该fixed_Layer的坐标意义：[a,b,c,d]代表[第a行，第b列，第c通道，第d个卷积核]
            # weight,bias = self.fixed_Layer.get_weights()
            weight = self.normal_weight
            bias = self.normal_bias

            index_list = self.location
            # print(index_list)
            bit_num = self.exp_location
            # print(weight.shape)
            faulty_weight = weight[index_list[0]-1,index_list[1]-1,index_list[2]-1,index_list[3]-1]
            # print(faulty_weight)
            weight[index_list[0]-1,index_list[1]-1,index_list[2]-1,index_list[3]-1] = inject_SBF(faulty_weight,bit_num)
            # print(weight[index_list[fixed_Layer][0],index_list[fixed_Layer][1],index_list[fixed_Layer][2],index_list[fixed_Layer][3]])
            weights = (weight,bias)
            self.normal_Layer.set_weights(weights)

        if self.cur_page==1:
            convs = self.normal_inject_convs
            for fixed_Layer in convs:
                weight,bias = fixed_Layer.get_weights()
                rate = self.rate/100

                faulty_weight = inject_layer_MBF(weight,rate)
                weights = (faulty_weight,bias)
                fixed_Layer.set_weights(weights)

        if self.cur_page==2:
            weight = self.normal_weight2
            bias = self.normal_bias2
            index_list = self.location2
            target = self.target

            weight[index_list[0]-1,index_list[1]-1,index_list[2]-1,index_list[3]-1] = target

            weights = (weight,bias)
            self.normal_Layer2.set_weights(weights)

    def fixed_inject_check(self):
        # 注入的时候判断当前是哪一页，并且根据判断读取对应页面参数。
        # 将页面参数存入info.fp中,因为有多个接口要用
        if self.cur_page == 0:
            self.layer = self.ui.sB_layer.value()
            self.location = [self.ui.sB_location1.value(),self.ui.sB_location2.value(),self.ui.sB_location3.value(),self.ui.sB_location4.value()]
            self.exp_location = self.ui.sB_exp_location.value()
        elif self.cur_page == 1:
            self.rate = self.ui.sB_rate.value()
        else:
            self.layer2 = self.ui.sB_layer2.value()
            self.location2 = [self.ui.sB_location2_1.value(),self.ui.sB_location2_2.value(),self.ui.sB_location2_3.value(),self.ui.sB_location2_4.value()]
            self.target = self.ui.sB_target.value()
        if self.check():
            # 注入
            self.fixed_inject()
            self.success= infoWindow("注入成功")
            self.success.ui.show()
        else:
            self.success= infoWindow("请确定参数范围，输入正确参数")
            self.success.ui.show()
    def fixed_inject(self):
        if self.cur_page==0:
            # 针对该fixed_Layer的坐标意义：[a,b,c,d]代表[第a行，第b列，第c通道，第d个卷积核]
            # weight,bias = self.fixed_Layer.get_weights()
            weight = self.weight
            bias = self.bias

            index_list = self.location
            # print(index_list)
            bit_num = self.exp_location
            # print(weight.shape)
            faulty_weight = weight[index_list[0]-1,index_list[1]-1,index_list[2]-1,index_list[3]-1]
            # print(faulty_weight)
            weight[index_list[0]-1,index_list[1]-1,index_list[2]-1,index_list[3]-1] = inject_SBF(faulty_weight,bit_num)
            # print(weight[index_list[fixed_Layer][0],index_list[fixed_Layer][1],index_list[fixed_Layer][2],index_list[fixed_Layer][3]])
            weights = (weight,bias)
            self.fixed_Layer.set_weights(weights)

        if self.cur_page==1:
            convs = self.fixed_inject_convs
            for fixed_Layer in convs:
                weight,bias = fixed_Layer.get_weights()
                rate = self.rate/100

                faulty_weight = inject_layer_MBF(weight,rate)
                weights = (faulty_weight,bias)
                fixed_Layer.set_weights(weights)

        if self.cur_page==2:
            weight = self.weight2
            bias = self.bias2
            index_list = self.location2
            target = self.target

            weight[index_list[0]-1,index_list[1]-1,index_list[2]-1,index_list[3]-1] = target

            weights = (weight,bias)
            self.fixed_Layer2.set_weights(weights)

    def check(self)->bool:
        if self.cur_page ==0:
            if self.layer not in range(1,self.info.layer_num+1):
                return False
            weight = self.weight
            # print(weight.shape)
            # compares = [len(weight),len(weight[0]),len(weight[0][0]),len(len(weight[0][0][0]))]
            # print(self.location)
            compares = weight.shape
            for i in range(4):
                # if compares[i]==1:
                    # print(self.location[i])
                    # print(range(0,compares[i]))
                if self.location[i] not in range(1,compares[i]+1):
                    return False
            if self.exp_location not in range(1,32):
                return False
            return True
        if self.cur_page ==1:
            if self.rate>60 or self.rate<=0: # 最大错误率只能60%
                return False
            return True
        if self.cur_page==2:
            if self.layer2 not in range(1,self.info.layer_num+1):
                return False
            weight = self.weight2
            # print(weight.shape)
            # compares = [len(weight),len(weight[0]),len(weight[0][0]),len(len(weight[0][0][0]))]
            compares = weight.shape
            # print(self.location2)
            # print(compares)
            for i in range(4):
                if self.location2[i] not in range(1,compares[i]+1):
                    # print(2)
                    return False
            if self.target<self.mintarget or self.target>=self.maxtarget:
                return False
            return True
        return False
