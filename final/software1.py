import re
import copy
import sys
import os
import time

from PIL import Image


import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch.utils.tensorboard import SummaryWriter

from my_final_ui import *
from entity.info import Info, infoWindow
import utils.pytorch.change_relu1 as cr
import utils.pytorch.find_gate1 as fg
from utils.keras.fault_inject import *
#   pytorch版本
import torch.nn as nn
import torch
import subprocess
from models.LeNet import LeNet, LeNet2
import torchsummary
from anytree import Node
from build_tree import makeTree, get_entity, get_leaf_name

import sys
from PySide2.QtWidgets import QApplication, QMainWindow, QFileDialog, QDialog
from PySide2.QtGui import QIcon
from PySide2.QtCore import *
from PySide2 import QtCore, QtGui, QtWidgets

from PySide2.QtWidgets import *
from PySide2.QtGui import QIntValidator, QDoubleValidator, QRegExpValidator
import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"  #（保证程序cuda序号与实际cuda序号对应）
os.environ['CUDA_VISIBLE_DEVICES'] = "0,1,2,3"  #（代表仅使用第0，1号GPU）



BACKGROUND_COLOR = "#222222"
TITLE_COLOR = '#D9821E'  # 橙色

# 黄色：'#FFFF00'

# 按钮高度
BUTTON_HEIGHT = 18
# 按钮宽度
BUTTON_WIDTH = 39
device = 'cuda'


class mwindow(QMainWindow, Ui_MainWindow):
    # 打包命令：pyinstaller main.py --noconsole --hidden-import PySide2.QtXml
    # 打包之后，需要手动将ui文件，放在dist文件夹中，路s径和load中的路径要一致（相对于exe文件）。（图片等静态资源文件同理）

    # 对于ui文件转为py文件，应先conda activate包含pyside2的环境，然后在ui文件所在目录下运行
    # pyside2-uic my_final_ui.ui > my_final_ui.py

    def __init__(self, model, input_shape):
        super(mwindow, self).__init__()
        self.data_dir = None
        self.thread = None
        self.prog = None
        self.model = model.to(device)
        self.setupUi(self)

        self.root = QFileInfo(__file__).absolutePath()
        self.setWindowIcon(QIcon(self.root+'/img/icon.png'))

        self.pushButton.clicked.connect(self.close)
        self.pushButton_2.clicked.connect(self.ButtonMinSlot)
        #结果查看按钮修改
        self.pushButton_4.clicked.connect(self.upload_origin_model)
        self.pushButton_5.clicked.connect(self.upload_fixed_model)
        self.pushButton_6.clicked.connect(self.upload_dataset)
        self.pushButton_7.clicked.connect(self.show_struct)
        self.pB_predict_origin_inject.clicked.connect(self.predict_origin_result)
        self.pB_predict_fixed_inject.clicked.connect(self.predict_fixed_result)

        self.setWindowFlag(QtCore.Qt.FramelessWindowHint)  # 去边框
        self.setAttribute(QtCore.Qt.WA_TranslucentBackground)  # 设置窗口背景透明
        self.centralwidget.setAttribute(QtCore.Qt.WA_TranslucentBackground)  # 设置pyqt自动生成的centralwidget背景透明
        self.centralwidget.setAutoFillBackground(True)
        self.pushButton.setFixedSize(QSize(BUTTON_WIDTH, BUTTON_HEIGHT))  # 设置按钮大小
        self.pushButton_2.setFixedSize(QSize(BUTTON_WIDTH, BUTTON_HEIGHT))  # 设置按钮大小

        Qss = 'QWidget#widget_2{background-color: %s;}' % BACKGROUND_COLOR
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
        self.setStyleSheet(Qss)  # 边框部分qss重载

        self.info = Info()

        # 浮点校验器 [-360,360]，精度：小数点后2位
        doubleValidator = QDoubleValidator(self)
        doubleValidator.setRange(0, 1.00000)
        doubleValidator.setNotation(QDoubleValidator.StandardNotation)  # 校准显示
        # 设置精度，小数点5位
        doubleValidator.setDecimals(5)
        self.iE_rate.setValidator(doubleValidator)

        doubleValidator2 = QDoubleValidator(self)
        doubleValidator2.setRange(0, 65535)
        doubleValidator2.setNotation(QDoubleValidator.StandardNotation)
        # 设置精度，小数点3位
        doubleValidator2.setDecimals(3)
        self.iE_target.setValidator(doubleValidator2)

        self.tabWidget.currentChanged.connect(self.tab_change)
        self.currentPage = 0
        # 所有trigger
        self.pB_upload_model.clicked.connect(self.upload_model)
        self.pB_upload_data.clicked.connect(self.upload_data)
        self.upload_page = 0  # 模型和数据集获取方式是一还是二
        self.tabWidget_upload.currentChanged.connect(self.upload_tab_change)
        # # 增强
        self.dataset = []
        self.pB_fix.clicked.connect(self.fix)
        self.pB_download_fixed.clicked.connect(self.download_fixed)
        # self.tB_ms.ensureCursorVisible()
        # self.tB_ms.document().setMaximumBlockCount(1000)
        # # 错误注入
        self.pB_inject_origin.clicked.connect(self.inject_origin)
        self.pB_inject_fixed.clicked.connect(self.inject_fixed)
        self.tabWidget_inject.currentChanged.connect(self.inject_tab_change)
        self.pB_download_origin_inject.clicked.connect(self.download_origin_inject)
        self.pB_download_fixed_inject.clicked.connect(self.download_fixed_inject)

        self.layers_name = []
        self.print_flag = 0
        self.input_shape = input_shape

        # 不需要点击选择模型，直接会调用upload_model显示模型信息
        self.upload_model()


    # 重定向输出函数，将输出定向到UI界面的textBox中
    def write(self, string):

        # string = string.split('-',1)[1]
        # string = re.sub(r'\s+', ' ', string)
        # string = re.sub(r'\[', '\t[', string)
        # string = re.sub(r']', ']\t', string)

        if self.print_flag == 0:
            self.tB_origin_model.append(string)
        elif self.print_flag == 1:
            self.tB_fixed_model.append(string)

    # 这个函数强行在模型输入之前不允许页面切换
    def tab_change(self, x):
        # 这里的x的0,1,2,3分别对应程序上边的各个按钮
        if x == 0:
            pass
        if x == 1:
            # self.infob包含原模型（注入模型），（卷积层），修改模型（注入模型），（卷积层）
            if self.info.model == None and self.upload_page == 0:
                print("请先上传模型")
                self.tabWidget.setCurrentIndex(0)  # 回到首页
        if x == 2:
            if self.info.fixed_model != None:
                # 参数
                self.inject_current_page = 0
                self.mintarget = -100000
                self.maxtarget = 100000
                # 方式一
                self.layer_1 = 0  # 从第一层开始。
                self.location_1 = [0, 0, 0, 0]
                self.exp_location = 32
                self.type_single = "置一"
                # 方式二
                self.layer_2 = 0
                self.rate = 0.1
                self.type_multiply = "置一"
                # 方式三
                self.layer_3 = 0
                self.location_2 = [0, 0, 0, 0]
                self.target = 99

                # 拷贝模型（保证每次回来都是在修复模型上重新错误注入）
                self.info.fixed_inject_model = clone_model(self.info.fixed_model)
                # weights = self.info.fixed_model.get_weights()
                # self.info.fixed_inject_model.set_weights(weights)
                root = Node('root')
                makeTree(self.info.fixed_inject_model, root, '')
                modules = get_leaf_name(self.info.fixed_inject_model, root)
                # 存储所有卷积层
                self.fixed_inject_convs = []

                for module in modules:
                    entity = get_entity(self.info.fixed_inject_model, module)
                    index = module.split('_')
                    index = index[len(index) - 1]
                    entity = getattr(entity, index)
                    if isinstance(entity, nn.Conv2d):
                        self.fixed_inject_convs.append(module)
                    elif isinstance(entity,torch.ao.nn.quantized.modules.conv.Conv2d):
                        self.fixed_inject_convs.append(module)
                # 这里把info.model的模型及权重赋给origin_inject_model
                self.info.origin_inject_model = clone_model(self.model)
                # weights = self.info.model.get_weights()
                # self.info.origin_inject_model.set_weights(weights)

                # 这里由于只保存convs名字，所以直接复制
                self.origin_inject_convs = self.fixed_inject_convs
                # for module in modules:
                #     entity = get_entity(self.info.origin_inject_model, module)
                #     index = module.split('_')
                #     index = index[len(index) - 1]
                #     entity = getattr(entity, index)
                #     if isinstance(entity, nn.Conv2d):
                #         self.origin_inject_convs.append(module)

                # 这里要得到weight,bias，传入对应model
                self.fixed_state_dict = self.info.fixed_inject_model.state_dict()
                self.origin_state_dict = self.info.origin_inject_model.state_dict()
                self.sB_layer_1.valueChanged.connect(self.layer_change_1)
                self.sB_layer_2.valueChanged.connect(self.layer_change_2)
                # 必须分开，不能在一个函数中用if页面来判断。会有逻辑错误。
                self.sB_layer_3.valueChanged.connect(self.layer_change_3)

                # 设置值(# ui里需要将所有值设置为0，后面修改，就必先走change函数，就必有self.weight和self.fixed_Layer)
                self.sB_layer_1.setValue(self.info.layer_num)
                self.sB_layer_2.setValue(self.info.layer_num)
                self.sB_layer_3.setValue(self.info.layer_num)
                # 设置最大值和最小值
                self.sB_layer_1.setRange(1, len(self.fixed_inject_convs))
                self.sB_layer_2.setRange(1, len(self.fixed_inject_convs))
                self.sB_layer_3.setRange(1, len(self.fixed_inject_convs))
                self.sB_exp_location.setRange(1, 32)
            elif self.info.fixed_model == None:
                print("请先修复")
                self.infoWindow = infoWindow("请先修复")
                self.infoWindow.ui.show()
                self.tabWidget.setCurrentIndex(1)

        if x == 3:
            if self.info.model == None:
                print("请先选择模型")
                self.infoWindow = infoWindow("请先选择模型")
                self.infoWindow.ui.show()
                self.tabWidget.setCurrentIndex(0)
            else:
                if self.info.fixed_model == None:
                    print("请先修复")
                    self.infoWindow = infoWindow("请先修复")
                    self.infoWindow.ui.show()
                    self.tabWidget.setCurrentIndex(1)
                else:
                    if self.info.fixed_inject_model == None:
                        print("请先错误注入")
                        self.infoWindow = infoWindow("确定不进行错误注入")
                        self.infoWindow.ui.show()
                        self.tabWidget.setCurrentIndex(2)
                    else:
                        print("进行精确度计算。")
                        self.lineEdit.setText("请选择原模型")
                        self.lineEdit_2.setText("请选择增强模型")
                        self.lineEdit_3.setText("请选择数据集")
                        # self.info.fixed_inject_model.compile(optimizer="rmsprop", loss='categorical_crossentropy',
                        #                                      metrics=["accuracy"])
                        # _, self.fixed_accurcy = self.info.fixed_inject_model.evaluate(self.x_test, self.y_test)
                        # print(self.fixed_accurcy)
                        # self.iE_fixed_result.clear()
                        # self.iE_fixed_result.setText(str(self.fixed_accurcy * 100) + "%")
                        #
                        # self.info.origin_inject_model.compile(optimizer="rmsprop", loss='categorical_crossentropy',
                        #                                       metrics=["accuracy"])
                        # _, self.origin_accurcy = self.info.origin_inject_model.evaluate(self.x_test, self.y_test)
                        # print(self.origin_accurcy)
                        # self.iE_origin_result.clear()
                        # self.iE_origin_result.setText(str(self.origin_accurcy * 100) + "%")

        self.currentPage = x

    # show
    def update_models_print(self, x):

        # 通过读取forward函数的定义，我们可以获取到输入张量形状的参数名x
        # 然后，访问模型的状态字典并通过参数名获取对应的张量形状
        # input_layer_name = self.model.forward.__code__.co_varnames[1]
        # input_size = self.model.state_dict()[input_layer_name].shape

        # 将输出重定向到自定义输出函数
        sys.stdout = self

        input_size = []
        input_size.append(self.input_shape[2])
        input_size.append(self.input_shape[0])
        input_size.append(self.input_shape[1])
        input_size = tuple(input_size)

        if x == 1:
            self.tB_origin_model.clear()
            self.print_flag = 0
            print(self.info.model)
            # torchsummary.summary(self.info.model, input_size=input_size)
        if x == 2:
            self.tB_fixed_model.clear()
            self.print_flag = 1

            # 这里输出会有奇怪的tensor信息，会在write重定向函数中过滤
            # torchsummary.summary(self.info.fixed_model, input_size=input_size)
            print(self.info.fixed_model)
        if x == 3:
            self.tB_origin_model.clear()
            self.print_flag = 0
            self.info.origin_inject_model.summary(print_fn=self.tB_origin_model.append)
        if x == 4:
            self.tB_fixed_model.clear()
            self.print_flag = 1
            # self.info.fixed_inject_model.summary(print_fn=self.tB_fixed_model.append)

        # 恢复默认的输出流
        sys.stdout = sys.__stdout__

    # main
    def upload_tab_change(self, x):
        self.upload_page = x

    def upload_origin_model(self):
        self.origin_model_dir = QFileDialog.getOpenFileName(QMainWindow(), "选择文件夹",filter='pth(*.pth)')[0]
        if self.origin_model_dir == '':
            pass
        else:
            self.lineEdit.setText(self.origin_model_dir)

    def upload_fixed_model(self):
        self.fixed_model_dir = QFileDialog.getOpenFileName(QMainWindow(), "选择文件夹",filter='pth(*.pth)')[0]
        if self.fixed_model_dir == '':
            pass
        else:
            self.lineEdit_2.setText(self.fixed_model_dir)

    def predict_origin_result(self):
        try:
            self.prog = ProgressBar()
            self.thread = FinalThread(inject_result, self.dataset_dir,self.origin_model_dir)
            self.thread.progress.connect(self.prog.progress.setValue)
            self.thread.start()
            self.prog.run()
            #等待线程返回结果
            while self.thread.isRunning():
                QApplication.processEvents()  # Allow the application to process events and remain responsive
            res=self.thread.get_res()
            self.iE_origin_result.setText(str(res))
        except:
            print("请选择数据集和原注入模型")
    def predict_fixed_result(self):
        try:
            self.prog = ProgressBar()
            self.thread = FinalThread(inject_result, self.dataset_dir, self.fixed_model_dir)
            self.thread.progress.connect(self.prog.progress.setValue)
            self.thread.start()
            self.prog.run()
            # 等待线程返回结果
            while self.thread.isRunning():
                QApplication.processEvents()  # Allow the application to process events and remain responsive
            f_res = self.thread.get_res()
            self.iE_fixed_result.setText(str(f_res))
        except:
            print("请选择数据集和增强注入模型")
    def upload_model(self):
        # file = QFileDialog.getOpenFileName(QMainWindow(), "选择模型", filter="model (*.pth)")

        self.iE_model_uri.setText('无需选择模型')
        self.info.model = clone_model(self.model)
        # self.info.model.load_state_dict(torch.load(file[0]))
        # length = len(self.info.model.layers)

        length = 0
        state_dict = self.info.model.state_dict().items()
        for name, value in state_dict:
            if 'weight' in name:
                length += 1
            name = list(name.split('.'))
            name = name[0]
            if name not in self.layers_name:
                self.layers_name.append(name)


        self.update_models_print(1)
        # 数据上传（fixed_model=None,每次模型上传后都应该将fixed_model置为none）
        self.info.fixed_model = None
        self.tB_fixed_model.clear()

    #上传验证数据集
    def upload_dataset(self):
        self.dataset_dir = QFileDialog.getExistingDirectory(QMainWindow(), "选择文件夹")
        if self.dataset_dir == '':
            pass
        else:
            self.lineEdit_3.setText(self.dataset_dir)

    def upload_data(self):
        # print("上传数据")
        self.data_dir = QFileDialog.getExistingDirectory(QMainWindow(), "选择文件夹")
        if self.data_dir == '':
            pass
        else:
            # print(FileDirectory)
            self.dataset = []
            self.iE_data_uri.setText(self.data_dir)
            # self.tB_uri.append(FileDirectory) # 追加的形式 

    #展示模型结构
    def show_struct(self):
        try:
            model=torch.load(self.fixed_model_dir)
            path = os.getcwd()
            file = 'run'
            logdir = os.path.join(path, file)
            show_graph(model,logdir)

            #创建子进程执行cmd命令
            p = subprocess.Popen(
                f'tensorboard --logdir="{logdir}"',
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                encoding='gb2312'
            )
            pid=p.pid #子进程号
            os.system(f'start chrome  http://localhost:6006/')
            time.sleep(10)
            #消灭子进程
            os.system(f'taskkill /pid {pid} /F')
        except:
            self.infoWindow = infoWindow("请先上传增强模型")
            self.infoWindow.ui.show()

    def load_data(self):

        # 读取文件夹中的图片
        files_list = []
        self.dataset = []
        files_list = os.listdir(self.data_dir)

        print(self.data_dir)
        for filename in files_list:
            if filename.endswith(("png")):
                absoluate_path = self.data_dir + "/" + filename
                f = Image.open(absoluate_path)  # 注意，这里不能用plt.imread()，那样会导致读不到通道，只能读取灰度图。
                shape = [1]
                size = []
                # for i in range(1, 4):
                #     shape.append(self.info.model.layers[0].input_shape[0][i])

                for i in range(0, 2):
                    size.append(self.input_shape[i])
                    shape.append(self.input_shape[i])
                shape.append(self.input_shape[2])

                # 拉伸
                # f = f.resize(size, Image.ANTIALIAS)
                f = f.resize(size, Image.LANCZOS)

                # 修改通道
                if shape[3] == 1:
                    # 说明是单通道
                    f = f.convert('L')
                elif shape[3] == 3:
                    f = f.convert('RGB')
                else:
                    print("模型输入通道数有问题")

                # 转化为np并reshape
                f = np.array(f)
                # 归一化
                f = f / 255.0
                # print("resize后的形状:"+str(f.shape))
                f = np.reshape(f, shape)

                # 将 np 转化为 tensor
                f = torch.tensor(f)

                # print(f.shape)
                self.dataset.append(f)
                # print("数据集长度：{}".format(len(self.dataset)))
                # print(self.dataset.shape)
        self.dataset_test = []
        # 划分10%的图片为测试集，向上取整。 实际上是不能这样做的。原因如下：
        # 1.标签存储形式的多样性，有可能是图片名称，有可能是xml，有可能是文档，如果让用户自行选择，一是工程量爆炸，二是还不如用户下载完模型后，用自己的代码评估性能来得方便。
        # 2.标签类型多样，如定位任务中就根本没有精确度，如果不用精确度用loss，则需要用户自行选择loss的计算方式。
        #   也就是说所有的loss都有嵌入进这个程序。工作量爆炸，而且单看loss也看不出什么，没有意义。
        # 3.各种标签不同代表着用户需要先输入标签的格式，程序才能读取。输错一点就读不了。对用户而言，就是本来自己一个检测loss或其他指标代码就完事，结果整的复杂化到爆炸。
        if self.dataset == []:
            self.infoWindow = infoWindow("图片需要为png格式")
            self.infoWindow.ui.show()

    def fix(self):
        if self.upload_page == 0:
            # 说明是方式一。
            if self.dataset == None or self.data_dir == '' or self.data_dir == None:
                self.infoWindow = infoWindow("请先上传png图片")
                self.infoWindow.ui.show()
            else:
                self.load_data()
                # 修复方法。
                method = self.cB_fix_method.currentText()
                if method == "BRelu" or method == "FRelu" or method =='Clipper':
                    self.fix_by_layer()
                if method == "ChRelu":
                    self.fix_by_channel()



    def fix_by_layer(self):
        # 计算上边界
        self.prog = ProgressBar()
        self.b_type=self.comboBox.currentText()
        self.thread = WorkThread(fg.calculate_boundary,self.dataset, self.info.model,self.b_type)
        self.thread.finished.connect(self.on_thread_layer_finished)
        self.thread.progress.connect(self.prog.progress.setValue)
        self.thread.start()
        self.prog.run()

    def fix_by_channel(self):
        # 计算上边界
        self.prog = ProgressBar()
        self.b_type = self.comboBox.currentText()
        self.thread = WorkThread(fg.calculate_ch_boundary, self.dataset, self.info.model,self.b_type)
        self.thread.finished.connect(self.on_thread_ch_finished)
        self.thread.progress.connect(self.prog.progress.setValue)
        self.thread.start()
        self.prog.run()

        # 将耗时事件单独开一个线程完成

    def on_thread_layer_finished(self):
        boundarys = self.thread.get_boundarys()

        if len(boundarys) == 0:
            self.fail = infoWindow("无法找到Relu或模型已经被修复")
            self.fail.ui.show()
            return

        # 根据上边界来fix
        # 根据按钮值不同，选择不同的容错方式

        print(self.cB_fix_method.currentText())
        self.info.fixed_model = cr.fix(copy.deepcopy(self.info.model), boundarys,self.cB_fix_method.currentText())

        # 弹窗打印修复成功
        self.success = infoWindow("修复成功")
        self.success.ui.show()
        # 显示修复成功后的结构
        self.update_models_print(2)

        # 将卷积层列表拷贝（useless）
        fixed_model_list = list(self.info.fixed_model.children())
        length = len(fixed_model_list)
        self.info.convs = []  # 清空原模型卷积层列表，节省空间,实际上根本没有用到这个变量。
        for i in range(length):
            if isinstance(fixed_model_list[i], nn.Conv2d) and isinstance(fixed_model_list[i + 1], nn.ReLU):
                self.info.fixed_convs.append(fixed_model_list[i])
        self.info.layer_num = len(self.info.fixed_convs)

    def on_thread_ch_finished(self):
        # 在主线程中处理任务完成后的操作
        # self.label.setText("Thread finished!")
        boundarys = self.thread.get_boundarys()

        if len(boundarys) == 0:
            self.fail = infoWindow("无法找到Relu或模型已经被修复")
            self.success.ui.show()
            return

            # 根据上边界来fix
            # print(boundarys)
        if boundarys == False:
            self.success = infoWindow("某层通道数超过5000")
            self.success.ui.show()
        else:
            self.info.fixed_model = cr.fix(copy.deepcopy(self.info.model), boundarys,self.cB_fix_method.currentText())

            # 弹窗打印修复成功
            self.success = infoWindow("修复成功")
            self.success.ui.show()
            # 显示修复成功后的结构
            self.update_models_print(2)

            # 将卷积层列表拷贝(useless)
            # fixed_model_list = list(self.info.fixed_model.children())
            # length = len(fixed_model_list)
            # self.info.convs = []  # 清空原模型卷积层列表，节省空间,实际上根本没有用到这个变量。
            # for i in range(length):
            #     if isinstance(fixed_model_list[i], nn.Conv2d) and isinstance(fixed_model_list[i + 1], nn.ReLU):
            #         self.info.fixed_convs.append(fixed_model_list[i])
            # self.info.layer_num = len(self.info.fixed_convs)

    def download_fixed(self):
        if self.info.fixed_model == None:
            self.success = infoWindow("请先修复")
            self.success.ui.show()
        else:
            path = QFileDialog.getExistingDirectory(QMainWindow(), "选择文件夹")
            if path != "":

                path = path + "/fixed_models.pth"
                torch.save(self.info.fixed_model, path)
                self.success = infoWindow("下载完成")
                self.success.ui.show()
            else:
                self.success = infoWindow("请选择下载路径")
                self.success.ui.show()

    # inject
    def download_origin_inject(self):
        if self.info.origin_inject_model == None:
            self.success = infoWindow("请先修复")
            self.success.ui.show()
        else:
            path = QFileDialog.getExistingDirectory(QMainWindow(), "选择文件夹")
            if path != "":

                path = path + "/origin_inject_models.pth"
                try:
                    torch.save(self.info.origin_inject_model, path)
                    self.success = infoWindow("下载完成")
                    self.success.ui.show()
                except:
                    print("输入路径错误，请检查是否为英文路径")
            else:
                self.success = infoWindow("请选择下载路径")
                self.success.ui.show()

    def download_fixed_inject(self):
        if self.info.fixed_inject_model == None:
            self.success = infoWindow("请先修复")
            self.success.ui.show()
        else:
            path = QFileDialog.getExistingDirectory(QMainWindow(), "选择文件夹")
            if path != "":
                path = path + "/fixed_inject_models.pth"
                try:
                    torch.save(self.info.fixed_inject_model, path)
                    self.success = infoWindow("下载完成")
                    self.success.ui.show()
                except:
                    print("输入路径错误，请检查是否为英文路径")
            else:
                self.success = infoWindow("请选择下载路径")
                self.success.ui.show()

    def inject_tab_change(self, x):
        # 根据当前页面注入错误
        self.inject_current_page = x

    def layer_change_1(self):
        # fixed_state_dict=fixed_model.state_dict()
        # origin_state_dict=origin_model.state_dict()
        # print("指定层的单比特注入")
        self.layer_1 = self.sB_layer_1.value()
        self.fixed_layer = self.fixed_inject_convs[self.layer_1 - 1]  # 因为convs下标从0开始，但用户输出最小值为1，所以拿的时候要-1
        self.fixed_layer = str(self.fixed_layer).replace('_', '.')  # 这里要索引字典，所以把_替换为.
        self.normal_layer = self.fixed_layer
        self.weight = self.fixed_state_dict[self.fixed_layer + '.weight']
        # 这里因为卷积层后面由bn层，所以判断是否存在bias
        try:
            self.bias = self.fixed_state_dict[self.fixed_layer + '.bias']
        except:
            print("卷积层没有bias")
        # self.weight, self.bias = self.fixed_layer.get_weights()  # (weight,bias) 才叫weights
        compares = self.weight.shape  # 两个shape都是一样的

        self.normal_weight = self.origin_state_dict[self.normal_layer + '.weight']
        # 这里因为卷积层后面由bn层，所以判断是否存在bias
        try:
            self.normal_bias = self.origin_state_dict[self.normal_layer + '.bias']
        except:
            pass
        # 设置范围(必须先设置范围)
        self.sB_location_1_1.setRange(1, compares[0])
        self.sB_location_1_2.setRange(1, compares[1])
        self.sB_location_1_3.setRange(1, compares[2])
        self.sB_location_1_4.setRange(1, compares[3])

        self.sB_location_1_1.setValue(compares[0])
        self.sB_location_1_2.setValue(compares[1])
        self.sB_location_1_3.setValue(compares[2])
        self.sB_location_1_4.setValue(compares[3])

    def layer_change_2(self):
        # print("指定层多比特注入")
        # fixed_state_dict = fixed_model.state_dict()
        # origin_state_dict = origin_model.state_dict()
        self.layer_2 = self.sB_layer_2.value()
        layer_number2 = self.layer_2 - 1
        self.fixed_layer2 = self.fixed_inject_convs[self.layer_2 - 1]  # 因为convs下标从0开始，但用户输出最小值为1，所以拿的时候要-1
        self.fixed_layer2 = str(self.fixed_layer2).replace('_', '.')  # 这里要索引字典，所以把_替换为.
        self.normal_layer_2 = self.fixed_layer2
        self.weight2 = self.fixed_state_dict[self.fixed_layer2 + '.weight']
        # 这里因为卷积层后面由bn层，所以判断是否存在bias
        try:
            self.bias2 = self.fixed_state_dict[self.fixed_layer2 + '.bias']
        except:
            pass
        self.normal_weight2 = self.origin_state_dict[self.normal_layer_2 + '.weight']
        # 这里因为卷积层后面由bn层，所以判断是否存在bias
        try:
            self.normal_bias2 = self.origin_state_dict[self.normal_layer_2 + '.bias']
        except:
            pass
        # print(self.layer2)
        # self.fixed_layer_2 = self.fixed_inject_convs[self.layer_2 - 1]
        # self.weight2, self.bias2 = self.fixed_layer_2.get_weights()
        #
        # self.normal_layer_2 = self.origin_inject_convs[self.layer_2 - 1]
        # self.normal_weight2, self.normal_bias2 = self.normal_layer_2.get_weights()

    def layer_change_3(self):
        # print("指定层的特定数修改")
        # self.fixed_state_dict = fixed_model.state_dict()
        # self.origin_state_dict = origin_model.state_dict()
        self.layer_3 = self.sB_layer_3.value()
        self.fixed_layer3 = self.fixed_inject_convs[self.layer_3 - 1]  # 因为convs下标从0开始，但用户输出最小值为1，所以拿的时候要-1
        self.fixed_layer3 = str(self.fixed_layer3).replace('_', '.')  # 这里要索引字典，所以把_替换为.
        self.normal_layer_3 = self.fixed_layer3
        self.weight3 = self.fixed_state_dict[self.fixed_layer3 + '.weight']
        # 这里因为卷积层后面由bn层，所以判断是否存在bias
        try:
            self.bias3 = self.fixed_state_dict[self.fixed_layer3 + '.bias']
        except:
            pass
        self.normal_weight3 = self.origin_state_dict[self.normal_layer_3 + '.weight']
        # 这里因为卷积层后面由bn层，所以判断是否存在bias
        try:
            self.normal_bias3 = self.origin_state_dict[self.normal_layer_3 + '.bias']
        except:
            pass
        compares = self.weight3.shape
        # self.layer_3 = self.sB_layer_3.value()
        # # print(self.layer2)
        # self.fixed_layer_3 = self.fixed_inject_convs[self.layer_3 - 1]
        # self.weight3, self.bias3 = self.fixed_layer_3.get_weights()
        # compares = self.weight3.shape
        #
        # self.normal_layer_3 = self.origin_inject_convs[self.layer_3 - 1]
        # self.normal_weight3, self.normal_bias3 = self.normal_layer_3.get_weights()

        # 设置范围
        self.sB_location_2_1.setRange(1, compares[0])
        self.sB_location_2_2.setRange(1, compares[1])
        self.sB_location_2_3.setRange(1, compares[2])
        self.sB_location_2_4.setRange(1, compares[3])

        self.sB_location_2_1.setValue(compares[0])
        self.sB_location_2_2.setValue(compares[1])
        self.sB_location_2_3.setValue(compares[2])
        self.sB_location_2_4.setValue(compares[3])

    def inject_origin(self):
        # 注入的时候判断当前是哪一页，并且根据判断读取对应页面参数。
        # 将页面参数存入info.fp中,因为有多个接口要用
        flag = True
        if self.inject_current_page == 0:
            self.layer_1 = self.sB_layer_1.value()
            self.location_1 = [self.sB_location_1_1.value(), self.sB_location_1_2.value(), self.sB_location_1_3.value(),
                               self.sB_location_1_4.value()]
            self.exp_location = self.sB_exp_location.value()
            self.type_single = self.cB_type_single.currentText()
        elif self.inject_current_page == 1:
            self.rate = self.iE_rate.text()  # 文本转小数。
            self.rate = float(self.rate)
            if self.rate > 1 or self.rate < 0:
                flag = False
            self.layer_2 = self.sB_layer_2.value()
            self.type_multiply = self.cB_type_multiply.currentText()
        else:
            self.layer_3 = self.sB_layer_3.value()
            self.location_2 = [self.sB_location_2_1.value(), self.sB_location_2_2.value(), self.sB_location_2_3.value(),
                               self.sB_location_2_4.value()]
            self.target = self.iE_target.text()
            self.target = float(self.target)
        if flag:
            # 注入
            self.origin_inject()
            self.success = infoWindow("注入成功")
            self.success.ui.show()
        else:
            self.success = infoWindow("请确定参数范围，输入正确参数")
            self.success.ui.show()

    def origin_inject(self):
        if self.inject_current_page == 0:
            # 针对该fixed_Layer的坐标意义：[a,b,c,d]代表[第a行，第b列，第c通道，第d个卷积核]
            # weight,bias = self.fixed_Layer.get_weights()
            weight = self.normal_weight

            index_list = self.location_1
            # print(index_list)
            bit_num = self.exp_location
            # print(weight.shape)
            faulty_weight = weight[index_list[0] - 1, index_list[1] - 1, index_list[2] - 1, index_list[3] - 1]
            # print(faulty_weight)
            weight[index_list[0] - 1, index_list[1] - 1, index_list[2] - 1, index_list[3] - 1] = inject_SBF(
                faulty_weight, bit_num)
            # 这里单比特注入有bug，注入过后对应位置的值仍然不变
            weight[index_list[0] - 1, index_list[1] - 1, index_list[2] - 1, index_list[3] - 1] = faulty_weight * 1e10
            # print(weight[index_list[fixed_Layer][0],index_list[fixed_Layer][1],index_list[fixed_Layer][2],index_list[fixed_Layer][3]])
            self.origin_state_dict[str(self.normal_layer) + '.weight'] = weight  # 这里state_dict是浅拷贝，所以下载模型的权重会相应改变
            # self.normal_layer.set_weights(weights)

        if self.inject_current_page == 1:
            convs = self.origin_inject_convs
            fixed_layer = convs[self.layer_2 - 1]
            fixed_layer = str(fixed_layer).replace('_', '.')  # 将fixed_layer名字格式转换
            weight = self.origin_state_dict[fixed_layer + '.weight']


            faulty_weight = inject_layer_MBF(weight, self.rate)
            self.origin_state_dict[fixed_layer + '.weight'] = faulty_weight  # 这里state_dict是浅拷贝，所以下载模型的权重会相应改变
            # fixed_layer.set_weights(weights)
        if self.inject_current_page == 2:
            weight = self.normal_weight2
            # bias = self.normal_bias2
            index_list = self.location_2
            target = self.target

            weight[index_list[0] - 1, index_list[1] - 1, index_list[2] - 1, index_list[3] - 1] = target
            # weights = (weight, bias)
            self.fixed_state_dict[
                str(self.normal_layer_2).replace('_', '.') + '.weight'] = weight  # 这里state_dict是浅拷贝，所以下载模型的权重会相应改变
            # self.normal_layer_2.set_weights(weights)

    def inject_fixed(self):
        # 注入的时候判断当前是哪一页，并且根据判断读取对应页面参数。
        # 将页面参数存入info.fp中,因为有多个接口要用
        flag = True
        if self.inject_current_page == 0:
            self.layer_1 = self.sB_layer_1.value()
            self.location_1 = [self.sB_location_1_1.value(), self.sB_location_1_2.value(), self.sB_location_1_3.value(),
                               self.sB_location_1_4.value()]
            self.exp_location = self.sB_exp_location.value()
            self.type_single = self.cB_type_single.currentText()
        elif self.inject_current_page == 1:
            self.rate = self.iE_rate.text()
            self.rate = float(self.rate)
            if self.rate < 0 or self.rate > 1:
                flag = False
            self.layer_2 = self.sB_layer_2.value()
            self.type_multiply = self.cB_type_multiply.currentText()  # 多比特错误方式。
        else:
            self.layer_3 = self.sB_layer_3.value()
            self.location_2 = [self.sB_location_2_1.value(), self.sB_location_2_2.value(), self.sB_location_2_3.value(),
                               self.sB_location_2_4.value()]
            self.target = self.iE_target.text()
            self.target = float(self.target)
        if flag:
            # 注入
            self.fixed_inject()
            self.success = infoWindow("注入成功")
            self.success.ui.show()
        else:
            self.success = infoWindow("请确定参数范围，输入正确参数")
            self.success.ui.show()

    def fixed_inject(self):
        if self.inject_current_page == 0:
            # 针对该fixed_Layer的坐标意义：[a,b,c,d]代表[第a行，第b列，第c通道，第d个卷积核]
            # weight,bias = self.fixed_Layer.get_weights()

            weight = self.weight
            # bias = self.bias

            index_list = self.location_1
            # print(index_list)
            bit_num = self.exp_location
            # print(weight.shape)
            faulty_weight = weight[index_list[0] - 1, index_list[1] - 1, index_list[2] - 1, index_list[3] - 1]
            # print(faulty_weight)
            weight[index_list[0] - 1, index_list[1] - 1, index_list[2] - 1, index_list[3] - 1] = inject_SBF(
                faulty_weight, bit_num)
            self.fixed_state_dict[str(self.normal_layer) + '.weight'] = weight
            # print(weight[index_list[fixed_Layer][0],index_list[fixed_Layer][1],index_list[fixed_Layer][2],index_list[fixed_Layer][3]])
            # weights = (weight, bias)
            # self.fixed_layer.set_weights(weights)

        if self.inject_current_page == 1:

            self.prog = ProgressBar()
            # (self, func, model, convs, layer_num, rate)
            self.thread = InjectThread(inject_with_valid,copy.deepcopy(self.info.fixed_model),
                                       self.fixed_inject_convs,self.layer_2,self.rate)
            self.thread.progress.connect(self.prog.progress.setValue)
            self.thread.start()
            self.prog.run()


            # 这里修改为在注入后就进行准确率验证，为耗时操作，因此采用多线程
            # 验证准确率
            # transform = transforms.Compose([
            #     transforms.Resize((224, 224)),
            #     transforms.ToTensor(),
            #     transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            # ])
            #
            # val_data = datasets.ImageFolder(root="/home/letian.chen/data/cat.dog/", transform=transform)
            # val_loader = torch.utils.data.DataLoader(val_data, batch_size=4, shuffle=False, num_workers=4)
            #
            # convs = self.fixed_inject_convs
            # fixed_layer = convs[self.layer_2 - 1]
            # fixed_layer = str(fixed_layer).replace('_', '.')  # 将fixed_layer名字格式转换
            # weight = self.fixed_state_dict[fixed_layer + '.weight']
            #
            # # 注入
            # faulty_weight = inject_layer_MBF(weight, self.rate)
            # self.fixed_state_dict[fixed_layer + '.weight'] = faulty_weight









        if self.inject_current_page == 2:
            weight = self.weight3
            index_list = self.location_2
            target = self.target

            weight[index_list[0] - 1, index_list[1] - 1, index_list[2] - 1, index_list[3] - 1] = target
            self.fixed_state_dict[str(self.normal_layer_2).replace('_', '.') + '.weight'] = weight

    # # 检查 iE_rate和iE_target是否符合规范。
    # def check()->bool:

    #     return False

    # def predict(self):
    #     if self.predict_img!="":
    #         origin_result = self.info.origin_inject_model.predict(self.predict_img)
    #         fixed_result = self.info.fixed_inject_model.predict(self.predict_img)
    #         self.iE_origin_result.setText(origin_result)
    #         self.iE_fixed_result.setText(fixed_result)
    #     else:
    #         self.success= infoWindow("请先上传图片")
    #         self.success.ui.show()
    # 重写鼠标点击移动，释放事件
    def mousePressEvent(self, event):
        if event.button() == QtCore.Qt.LeftButton:
            self.m_flag = True
            self.m_Position = event.globalPos() - self.pos()  # 获取鼠标相对窗口的位置
            event.accept()
            self.setCursor(QCursor(QtCore.Qt.OpenHandCursor))

    def mouseReleaseEvent(self, event):
        if event.button() == QtCore.Qt.LeftButton:
            self.m_flag = False
            self.setCursor(QCursor(QtCore.Qt.ArrowCursor))

    def mouseMoveEvent(self, event):
        if QtCore.Qt.LeftButton and self.m_flag:
            self.move(event.globalPos() - self.m_Position)  # 更改窗口位置
            event.accept()

    def ButtonMinSlot(self):
        self.showMinimized()

def fixModel(model, input_shape):
    app = QApplication(sys.argv)
    w = mwindow(model, input_shape)
    with open('qss/DarkOrangeQss.txt') as file:
        str1 = file.readlines()
        str1 = ''.join(str1).strip('\n')
        app.setStyleSheet(str1)  # 设置页面样式
    w.show()
    sys.exit(app.exec_())

# 自定义方法
def to_categorical(y, num_classes):
    """ 1-hot encodes a tensor """
    return np.eye(num_classes, dtype='uint8')[y]

def show_graph(model,dir):
    logdir = dir
    for i in os.listdir(logdir):
        os.remove(os.path.join(logdir, i))
    writer = SummaryWriter(logdir)
    writer.add_graph(model, input_to_model=torch.rand(4, 3, 224, 224).to(device))

def inject_result(data_dir,model,progress_callback):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    val_data = datasets.ImageFolder(root=data_dir, transform=transform)
    val_loader = torch.utils.data.DataLoader(val_data, batch_size=4, shuffle=False, num_workers=4)

    #progress_Callback
    total_num=len(val_loader)
    count=0

    model=torch.load(model)
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, labels in val_loader:
            count += 1
            progress = int(100 * count / total_num)
            progress_callback(progress)

            inputs, labels = inputs.to(device), labels.to(device)

            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    accuracy = correct / total * 100
    return accuracy

def clone_model(model, clone_weights=True):
    """
    Clones a PyTorch model.

    Parameters:
        model: the model to clone
        clone_weights: whether to clone the weights, default is True

    Returns:
        A cloned model instance.
    """
    model_clone = copy.deepcopy(model)

    # 如果不需要复制权重，重新初始化权重
    if not clone_weights:
        model_clone.apply(weight_reset)

    return model_clone


def weight_reset(m):
    """
    Resets the weights of the model.

    Parameters:
        m: model/module instance
    """
    reset_parameters = getattr(m, "reset_parameters", None)
    if callable(reset_parameters):
        m.reset_parameters()

class FinalThread(QThread):

    # 采用信号-槽机制，其中progress是信号，用于释放进度值信号然后被主线程接收
    # 其中func的参数列表应该包含一个回调函数progress_callback的地址
    # 从而在耗时函数中不断调用progress_callback
    # 进而进行self.progress.emit(value)

    progress = Signal(int)


    def __init__(self, func, data_dir,model):
        super().__init__()
        self.func = func

        self.model = model
        self.progress_value = 0
        self.data_dir=data_dir


    def run(self):
        self.origin_res= self.func(self.data_dir,self.model,self.progress_callback)

    def get_res(self):
        return self.origin_res

    def progress_callback(self, value):
        self.progress.emit(value)

class InjectThread(QThread):

    # 采用信号-槽机制，其中progress是信号，用于释放进度值信号然后被主线程接收
    # 其中func的参数列表应该包含一个回调函数progress_callback的地址
    # 从而在耗时函数中不断调用progress_callback
    # 进而进行self.progress.emit(value)

    progress = Signal(int)


    def __init__(self, func, model,convs,layer_num,rate):
        super().__init__()
        self.func = func

        self.model = model
        self.convs = convs
        self.progress_value = 0
        self.layer_num = layer_num
        self.rate = rate

    def run(self):
        self.func(self.model,self.convs,self.layer_num,self.rate,self.progress_callback)


    def progress_callback(self, value):
        self.progress.emit(value)

def get_weight(convs,fixed_state_dict):
    weights=[]
    for i in range(len(convs)):
        fixed_layer = convs[i]
        fixed_layer = str(fixed_layer).replace('_', '.')  # 将fixed_layer名字格式转换
        weight = copy.deepcopy(fixed_state_dict[fixed_layer + '.weight'])
        weights.append((fixed_layer,weight))
    return weights

def inject_with_valid(Tmodel,fixed_inject_convs,layer_num,rate,progress_callback):

        model = copy.deepcopy(Tmodel)

        # 这里修改为在注入后就进行准确率验证，为耗时操作，因此采用多线程
        # 验证准确率
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        fixed_state_dict = copy.deepcopy(model.state_dict())



        convs = fixed_inject_convs
        weights=get_weight(convs, fixed_state_dict)
        results = []

        # 注入50次，并验证准确率
        for i in range(10):

            val_data = datasets.ImageFolder(root=r"D:\容错\catsdogs\test", transform=transform)
            val_loader = torch.utils.data.DataLoader(val_data, batch_size=4, shuffle=False, num_workers=4)

            for fixed_layer,weight in weights:
                faulty_weight = inject_layer_MBF(copy.deepcopy(weight), 1e-6)
                diff_count = torch.sum(faulty_weight != weight).item()
                print(f"Number of different elements: {diff_count}")
                fixed_state_dict[fixed_layer + '.weight'] = faulty_weight

            model.load_state_dict(fixed_state_dict)


            model.eval()
            correct = 0
            total = 0
            with torch.no_grad():
                for inputs, labels in val_loader:

                    inputs, labels = inputs.to(device), labels.to(device)

                    outputs = model(inputs)
                    _, predicted = torch.max(outputs.data, 1)
                    total += labels.size(0)
                    correct += (predicted == labels).sum().item()

            accuracy = correct / total * 100
            results.append(accuracy)
            print(f'epoch : {i} Validation Accuracy: {accuracy:.2f}%')
            progress_callback((i+1) * 10)
        print(results)
        return




class ProgressBar(QObject):  # 进度条类
    def __init__(self):
        super(ProgressBar, self).__init__()
        self.progress = QProgressDialog('', '', 0, 0,)
        self.progress.setFixedSize(400, 200)
        self.progress.setWindowTitle('处理中')
        self.progress.setLabelText('当前进度值')
        self.progress.setCancelButtonText('取消')
        self.progress.setRange(0, 100)
        # self.progress.canceled.connect(lambda: print('进度对话框被取消'))

        self.progress.canceled.connect(self.cancel)
        self.progress.setAutoClose(True)  # value为最大值时自动关闭

    def run(self):  # 重写run，为了第二次启动时初始化做准备
        self.progress.show()
        self.progress.setValue(0)

    def cancel(self):  # 重写run，为了第二次启动时初始化做准备
        self.progress.close()
        self.progress.show()


# MyThread类允许传入不同的func函数地址
# 对不同容错方式，采用多线程方式进行寻找boundary的耗时操作
class WorkThread(QThread):

    # 采用信号-槽机制，其中progress是信号，用于释放进度值信号然后被主线程接收
    # 其中func的参数列表应该包含一个回调函数progress_callback的地址
    # 从而在耗时函数中不断调用progress_callback
    # 进而进行self.progress.emit(value)

    progress = Signal(int)
    finish = Signal()

    def __init__(self, func, dataset, model,b_type):
        super().__init__()
        self.func = func
        self.b_type=b_type
        self.dataset = dataset
        self.model = model
        self.boundarys = []
        self.progress_value = 0

    def run(self):
        self.boundarys = self.func(self.dataset, self.model,self.b_type,self.progress_callback)
        self.finish.emit()

    def progress_callback(self, value):
        self.progress.emit(value)

    def get_boundarys(self):
        return self.boundarys
