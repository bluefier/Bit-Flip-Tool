from PySide2.QtWidgets import QApplication, QMainWindow, QFileDialog,QDialog
from PySide2.QtUiTools import QUiLoader
from PySide2.QtGui import QIcon
from PySide2.QtCore import *
from PySide2 import QtCore, QtGui, QtWidgets
# inject
def inject_tab_change(self,x):
    # 根据当前页面注入错误
    self.inject_current_page = x
    
def layer_change_1(self):
    print("指定层的单比特注入")
    self.layer_1 = self.ui.sB_layer_1.value()
    # print(self.layer)
    self.fixed_layer = self.fixed_inject_convs[self.layer_1-1] # 因为convs下标从0开始，但用户输出最小值为1，所以拿的时候要-1
    self.weight,self.bias = self.fixed_layer.get_weights() # (weight,bias) 才叫weight
    compares = self.weight.shape # 两个shape都是一样的

    self.normal_layer = self.origin_inject_convs[self.layer_1-1]
    self.normal_weight,self.normal_bias = self.normal_layer.get_weights()

    # 设置范围(必须先设置范围)
    self.ui.sB_location_1_1.setRange(1,compares[0])
    self.ui.sB_location_1_2.setRange(1,compares[1])
    self.ui.sB_location_1_3.setRange(1,compares[2])
    self.ui.sB_location_1_4.setRange(1,compares[3])

    self.ui.sB_location_1_1.setValue(compares[0])
    self.ui.sB_location_1_2.setValue(compares[1])
    self.ui.sB_location_1_3.setValue(compares[2])
    self.ui.sB_location_1_4.setValue(compares[3])

def layer_change_2(self):
    print("指定层多比特注入")
    self.layer_2 = self.ui.sB_layer_2.value()
    # print(self.layer2)
    self.fixed_layer_2 = self.fixed_inject_convs[self.layer_2-1]
    self.weight2,self.bias2 = self.fixed_layer_2.get_weights()

    self.normal_layer_2 = self.origin_inject_convs[self.layer_2-1]
    self.normal_weight2,self.normal_bias2 = self.normal_layer_3.get_weights()

def layer_change_3(self):
    print("指定层的特定数修改")
    self.layer_3 = self.ui.sB_layer_3.value()
    # print(self.layer2)
    self.fixed_layer_3 = self.fixed_inject_convs[self.layer_3-1]
    self.weight3,self.bias3 = self.fixed_layer_3.get_weights()
    compares = self.weight3.shape

    self.normal_layer_3 = self.origin_inject_convs[self.layer_3-1]
    self.normal_weight3,self.normal_bias3 = self.normal_layer_3.get_weights()

    # 设置范围
    self.ui.sB_location2_1.setRange(1,compares[0])
    self.ui.sB_location2_2.setRange(1,compares[1])
    self.ui.sB_location2_3.setRange(1,compares[2])
    self.ui.sB_location2_4.setRange(1,compares[3])

    self.ui.sB_location2_1.setValue(compares[0])
    self.ui.sB_location2_2.setValue(compares[1])
    self.ui.sB_location2_3.setValue(compares[2])
    self.ui.sB_location2_4.setValue(compares[3])

def inject_origin(self):
    # 注入的时候判断当前是哪一页，并且根据判断读取对应页面参数。
    # 将页面参数存入info.fp中,因为有多个接口要用
    if self.inject_current_page == 0:
        self.layer_1 = self.ui.sB_layer_1.value()
        self.location_1 = [self.ui.sB_location_1_1.value(),self.ui.sB_location_1_2.value(),self.ui.sB_location_1_3.value(),self.ui.sB_location_1_4.value()]
        self.exp_location = self.ui.sB_exp_location.value()
    elif self.inject_current_page == 1:
        self.rate = self.ui.sB_rate.value()
        self.layer_2 = self.ui.sB_layer_2.value()
        self.type = self.ui.cB_type.value()
    else:
        self.layer_3 = self.ui.sB_layer_3.value()
        self.location_2 = [self.ui.sB_location_2_1.value(),self.ui.sB_location_2_2.value(),self.ui.sB_location_2_3.value(),self.ui.sB_location_2_4.value()]
        self.target = self.ui.sB_target.value()
    if True:
        # 注入
        self.origin_inject()
        self.success= infoWindow("注入成功")
        self.success.ui.show()
    else:
        self.success= infoWindow("请确定参数范围，输入正确参数")
        self.success.ui.show()

def origin_inject(self):
    if self.inject_current_page==0:
        # 针对该fixed_Layer的坐标意义：[a,b,c,d]代表[第a行，第b列，第c通道，第d个卷积核]
        # weight,bias = self.fixed_Layer.get_weights()
        weight = self.normal_weight
        bias = self.normal_bias

        index_list = self.location_1
        # print(index_list)
        bit_num = self.exp_location
        # print(weight.shape)
        faulty_weight = weight[index_list[0]-1,index_list[1]-1,index_list[2]-1,index_list[3]-1]
        # print(faulty_weight)
        weight[index_list[0]-1,index_list[1]-1,index_list[2]-1,index_list[3]-1] = inject_SBF(faulty_weight,bit_num)
        # print(weight[index_list[fixed_Layer][0],index_list[fixed_Layer][1],index_list[fixed_Layer][2],index_list[fixed_Layer][3]])
        weights = (weight,bias)
        self.normal_layer.set_weights(weights)

    if self.inject_current_page==1:
        convs = self.origin_inject_convs
        fixed_layer = convs[self.layer_2]
        
        weight,bias = fixed_layer.get_weights()
        rate = self.rate/100

        faulty_weight = inject_layer_MBF(weight,rate)
        weights = (faulty_weight,bias)
        fixed_layer.set_weights(weights)

    if self.inject_current_page==2:
        weight = self.normal_weight2
        bias = self.normal_bias2
        index_list = self.location2
        target = self.target

        weight[index_list[0]-1,index_list[1]-1,index_list[2]-1,index_list[3]-1] = target

        weights = (weight,bias)
        self.normal_layer_2.set_weights(weights)
def inject_fixed(self):
    # 注入的时候判断当前是哪一页，并且根据判断读取对应页面参数。
    # 将页面参数存入info.fp中,因为有多个接口要用
    if self.inject_current_page == 0:
        self.layer_1 = self.ui.sB_layer_1.value()
        self.location_1 = [self.ui.sB_location_1_1.value(),self.ui.sB_location_1_2.value(),self.ui.sB_location_1_3.value(),self.ui.sB_location_1_4.value()]
        self.exp_location = self.ui.sB_exp_location.value()
    elif self.inject_current_page == 1:
        self.rate = self.ui.sB_rate.value()
        self.layer_2 = self.ui.sB_layer_2.value()
        self.type = self.ui.cB_type.value()
    else:
        self.layer_3 = self.ui.sB_layer_3.value()
        self.location_2 = [self.ui.sB_location_2_1.value(),self.ui.sB_location_2_2.value(),self.ui.sB_location_2_3.value(),self.ui.sB_location_2_4.value()]
        self.target = self.ui.sB_target.value()
    if True:
        # 注入
        self.fixed_inject()
        self.success= infoWindow("注入成功")
        self.success.ui.show()
    else:
        self.success= infoWindow("请确定参数范围，输入正确参数")
        self.success.ui.show()
def fixed_inject(self):
    if self.inject_current_page==0:
        # 针对该fixed_Layer的坐标意义：[a,b,c,d]代表[第a行，第b列，第c通道，第d个卷积核]
        # weight,bias = self.fixed_Layer.get_weights()
        weight = self.weight
        bias = self.bias

        index_list = self.location_1
        # print(index_list)
        bit_num = self.exp_location
        # print(weight.shape)
        faulty_weight = weight[index_list[0]-1,index_list[1]-1,index_list[2]-1,index_list[3]-1]
        # print(faulty_weight)
        weight[index_list[0]-1,index_list[1]-1,index_list[2]-1,index_list[3]-1] = inject_SBF(faulty_weight,bit_num)
        # print(weight[index_list[fixed_Layer][0],index_list[fixed_Layer][1],index_list[fixed_Layer][2],index_list[fixed_Layer][3]])
        weights = (weight,bias)
        self.fixed_layer.set_weights(weights)

    if self.inject_current_page==1:
        convs = self.fixed_inject_convs
        fixed_layer = convs[self.layer_2]
        
        weight,bias = fixed_layer.get_weights()
        rate = self.rate/100

        faulty_weight = inject_layer_MBF(weight,rate)
        weights = (faulty_weight,bias)
        fixed_layer.set_weights(weights)

    if self.inject_current_page==2:
        weight = self.weight3
        bias = self.bias3
        index_list = self.location_2
        target = self.target

        weight[index_list[0]-1,index_list[1]-1,index_list[2]-1,index_list[3]-1] = target

        weights = (weight,bias)
        self.fixed_layer_3.set_weights(weights)

# 检查 iE_rate和iE_target是否符合规范。
def check()->bool:

    return False