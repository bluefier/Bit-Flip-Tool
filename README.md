# 神经网络容错软件

该软件由Python开发,GUI采用Pyside2多线程实现全部功能。通过对Relu层添加上界，进行错误截断，从而当出现比特翻转错误时有效防止精度下降，目前支持对pytorch模型进行容错操作。

##讲解视频
[https://www.bilibili.com/video/BV15N4y1n7YJ/?spm_id_from=333.999.0.0](https://www.bilibili.com/video/BV15N4y1n7YJ/?spm_id_from=333.999.0.0 "b站链接")
## 目录

- [神经网络容错软件](#神经网络容错软件)
  - [目录](#目录)
  - [核心文件介绍](#核心文件介绍)
  - [开发工具链](#开发工具链)
  - [设计思路](#设计思路)
  - [功能特性](#功能特性)
  - [安装](#安装)

## 核心文件介绍

- 根目录 pyqt ：虽然它叫pyqt，但实际上界面GUI采用的是Pyside2编写而成，混用pyqt与Pyside2很可能会导致软件报错。
- ui 文件夹 ：存放 UI 文件，其中 my_final_ui.ui 对应软件 GUI 界面，但需要通过以下指令将 ui 文件转换为 py 文件才能使用。
```bash
pyside2-uic my_final_ui.ui > my_final_ui.py
```
- final文件夹 ：项目代码实际存放位置
  - userFile.py : 软件运行入口
  - software1.py : QMainWindow核心框架，对容错功能进行实现，绝大部份功能代码都位于该文件中。
  - utils 文件夹 : 工具类，包括模型结构树构建，上界计算，ReLU替换等代码,使用pycharm第一次构建项目时需要右键点击utils.pytorch文件夹 [mark as source root] 将其标记为原目录


## 开发工具链

- 开发语言 ：Python 3.10
- GUI设计 ：QtCreator
- IDE : PyCharm
- 远程客户端连接 : MobaXterm

## 设计思路

- 关于软件接口设计 ： 其实选择在userFile.py中调用fixModel函数启动GUI是一个无奈的选择。不同于Keras，pytorch作为动态模型，在通过torch.load加载pth文件时需要获得模型的定义类进行序列化，从而构建模型。因此其实可以看到在userFile.py中包含了很多模型类，用于支持torch.load操作。如果不采用函数式API，很难获取模型定义类来实例化model。

- 关于多线程 ：我们应该确保GUI主线程只进行界面更新等简单操作，耗时操作应该使用QThread创建子线程完成，防止主界面停止响应。Pyside2提供了信号-槽函数的机制，其中子线程通过 Signal()注册信号，然后通过emit()释放信号，由主线程捕获。主线程捕获后通过connect()连接到槽函数，负责处理后续事件。


## 功能特性

该框架由四个子模块构成：数据上传，容错选择，错误注入，结果查看。

- 数据上传
- 容错选择
- 错误注入

```bash
#每个卷积层的参数已经保存在fixed_inject_convs[ ]数组中
#并且在错误注入页面拿到坐标参数，例如sB_layer_2 ---- 层索引

#针对该fixed_Layer的坐标意义：[a,b,c,d]代表[第a行，第b列，第c通道，第d个卷积核]
self.location_1 = [self.sB_location_1_1.value(), self.sB_location_1_2.value(), self.sB_location_1_3.value(),self.sB_location_1_4.value()]

index_list = self.location_1

#获得权重参数坐标
faulty_weight = weight[index_list[0] - 1, index_list[1] - 1, index_list[2] - 1, index_list[3] - 1]

bit_num = self.exp_location
#即1-32位置浮点数错误位置
inject_SBF(faulty_weight, bit_num)

```

## 安装

推荐在Linux上运行该项目，当克隆完成，确保已安装Anaconda后，可通过以下指令导入yml文件完成环境搭建。

```bash
# 示例安装步骤
git clone https://gitee.com/hedana/pyqt.git
cd pyqt
conda env create -f environment.yml
```
