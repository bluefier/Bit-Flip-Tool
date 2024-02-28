# -*- coding: utf-8 -*-

################################################################################
## Form generated from reading UI file 'final_UI.ui'
##
## Created by: Qt User Interface Compiler version 5.15.2
##
## WARNING! All changes made in this file will be lost when recompiling UI file!
################################################################################

from PySide2.QtCore import *
from PySide2.QtGui import *
from PySide2.QtWidgets import *


class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        if not MainWindow.objectName():
            MainWindow.setObjectName(u"MainWindow")
        MainWindow.resize(951, 800)
        MainWindow.setStyleSheet(u"")
        self.centralwidget = QWidget(MainWindow)
        self.centralwidget.setObjectName(u"centralwidget")
        self.widget = QWidget(self.centralwidget)
        self.widget.setObjectName(u"widget")
        self.widget.setGeometry(QRect(30, 19, 781, 21))
        sizePolicy = QSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.widget.sizePolicy().hasHeightForWidth())
        self.widget.setSizePolicy(sizePolicy)
        self.pushButton_2 = QPushButton(self.widget)
        self.pushButton_2.setObjectName(u"pushButton_2")
        self.pushButton_2.setGeometry(QRect(700, 0, 31, 21))
        self.pushButton = QPushButton(self.widget)
        self.pushButton.setObjectName(u"pushButton")
        self.pushButton.setGeometry(QRect(740, 0, 31, 21))
        self.pushButton_3 = QPushButton(self.widget)
        self.pushButton_3.setObjectName(u"pushButton_3")
        self.pushButton_3.setGeometry(QRect(0, 0, 31, 21))
        icon = QIcon()
        icon.addFile(u"img/btn_set_normal.png", QSize(), QIcon.Normal, QIcon.Off)
        self.pushButton_3.setIcon(icon)
        self.label = QLabel(self.widget)
        self.label.setObjectName(u"label")
        self.label.setGeometry(QRect(30, 0, 230, 21))
        self.label.setStyleSheet(u"")
        self.widget_3 = QWidget(self.centralwidget)
        self.widget_3.setObjectName(u"widget_3")
        self.widget_3.setGeometry(QRect(30, 690, 781, 16))
        self.widget_2 = QWidget(self.centralwidget)
        self.widget_2.setObjectName(u"widget_2")
        self.widget_2.setGeometry(QRect(30, 40, 781, 651))
        self.groupBox_2 = QGroupBox(self.widget_2)
        self.groupBox_2.setObjectName(u"groupBox_2")
        self.groupBox_2.setGeometry(QRect(0, 360, 781, 291))
        self.groupBox_2.setStyleSheet(u"")
        self.horizontalLayout = QHBoxLayout(self.groupBox_2)
        self.horizontalLayout.setObjectName(u"horizontalLayout")
        self.verticalLayout = QVBoxLayout()
        self.verticalLayout.setObjectName(u"verticalLayout")
        self.label12 = QLabel(self.groupBox_2)
        self.label12.setObjectName(u"label12")

        self.verticalLayout.addWidget(self.label12)

        self.tB_origin_model = QTextBrowser(self.groupBox_2)
        self.tB_origin_model.setObjectName(u"tB_origin_model")

        self.verticalLayout.addWidget(self.tB_origin_model)


        self.horizontalLayout.addLayout(self.verticalLayout)

        self.verticalLayout_3 = QVBoxLayout()
        self.verticalLayout_3.setObjectName(u"verticalLayout_3")
        self.label_2 = QLabel(self.groupBox_2)
        self.label_2.setObjectName(u"label_2")

        self.verticalLayout_3.addWidget(self.label_2)

        self.tB_fixed_model = QTextBrowser(self.groupBox_2)
        self.tB_fixed_model.setObjectName(u"tB_fixed_model")

        self.verticalLayout_3.addWidget(self.tB_fixed_model)


        self.horizontalLayout.addLayout(self.verticalLayout_3)

        self.tabWidget = QTabWidget(self.widget_2)
        self.tabWidget.setObjectName(u"tabWidget")
        self.tabWidget.setEnabled(True)
        self.tabWidget.setGeometry(QRect(0, 0, 781, 359))
        self.tabWidget.setCursor(QCursor(Qt.ArrowCursor))
        self.tabWidget.setIconSize(QSize(100, 20))
        self.tab_1 = QWidget()
        self.tab_1.setObjectName(u"tab_1")
        self.verticalLayout_5 = QVBoxLayout(self.tab_1)
        self.verticalLayout_5.setObjectName(u"verticalLayout_5")
        self.tabWidget_upload = QTabWidget(self.tab_1)
        self.tabWidget_upload.setObjectName(u"tabWidget_upload")
        self.tab_7 = QWidget()
        self.tab_7.setObjectName(u"tab_7")
        self.verticalLayout_4 = QVBoxLayout(self.tab_7)
        self.verticalLayout_4.setObjectName(u"verticalLayout_4")
        self.horizontalLayout_4 = QHBoxLayout()
        self.horizontalLayout_4.setObjectName(u"horizontalLayout_4")
        self.label_4 = QLabel(self.tab_7)
        self.label_4.setObjectName(u"label_4")

        self.horizontalLayout_4.addWidget(self.label_4)

        self.pB_upload_model = QPushButton(self.tab_7)
        self.pB_upload_model.setObjectName(u"pB_upload_model")

        self.horizontalLayout_4.addWidget(self.pB_upload_model)

        self.iE_model_uri = QLineEdit(self.tab_7)
        self.iE_model_uri.setObjectName(u"iE_model_uri")
        self.iE_model_uri.setReadOnly(False)

        self.horizontalLayout_4.addWidget(self.iE_model_uri)


        self.verticalLayout_4.addLayout(self.horizontalLayout_4)

        self.horizontalLayout_6 = QHBoxLayout()
        self.horizontalLayout_6.setObjectName(u"horizontalLayout_6")
        self.label_5 = QLabel(self.tab_7)
        self.label_5.setObjectName(u"label_5")

        self.horizontalLayout_6.addWidget(self.label_5)

        self.pB_upload_data = QPushButton(self.tab_7)
        self.pB_upload_data.setObjectName(u"pB_upload_data")

        self.horizontalLayout_6.addWidget(self.pB_upload_data)

        self.iE_data_uri = QLineEdit(self.tab_7)
        self.iE_data_uri.setObjectName(u"iE_data_uri")
        self.iE_data_uri.setReadOnly(False)

        self.horizontalLayout_6.addWidget(self.iE_data_uri)


        self.verticalLayout_4.addLayout(self.horizontalLayout_6)

        self.tabWidget_upload.addTab(self.tab_7, "")
        self.tab_8 = QWidget()
        self.tab_8.setObjectName(u"tab_8")
        self.verticalLayout_11 = QVBoxLayout(self.tab_8)
        self.verticalLayout_11.setObjectName(u"verticalLayout_11")
        self.horizontalLayout_21 = QHBoxLayout()
        self.horizontalLayout_21.setObjectName(u"horizontalLayout_21")
        self.label_17 = QLabel(self.tab_8)
        self.label_17.setObjectName(u"label_17")

        self.horizontalLayout_21.addWidget(self.label_17)

        self.cB_choose_model = QComboBox(self.tab_8)
        self.cB_choose_model.addItem("")
        self.cB_choose_model.addItem("")
        self.cB_choose_model.addItem("")
        self.cB_choose_model.setObjectName(u"cB_choose_model")

        self.horizontalLayout_21.addWidget(self.cB_choose_model)

        self.horizontalSpacer_10 = QSpacerItem(40, 20, QSizePolicy.Expanding, QSizePolicy.Minimum)

        self.horizontalLayout_21.addItem(self.horizontalSpacer_10)

        self.horizontalLayout_21.setStretch(0, 1)
        self.horizontalLayout_21.setStretch(1, 2)
        self.horizontalLayout_21.setStretch(2, 2)

        self.verticalLayout_11.addLayout(self.horizontalLayout_21)

        self.horizontalLayout_22 = QHBoxLayout()
        self.horizontalLayout_22.setObjectName(u"horizontalLayout_22")
        self.label_18 = QLabel(self.tab_8)
        self.label_18.setObjectName(u"label_18")

        self.horizontalLayout_22.addWidget(self.label_18)

        self.cB_choose_data = QComboBox(self.tab_8)
        self.cB_choose_data.addItem("")
        self.cB_choose_data.addItem("")
        self.cB_choose_data.setObjectName(u"cB_choose_data")

        self.horizontalLayout_22.addWidget(self.cB_choose_data)

        self.horizontalSpacer_11 = QSpacerItem(40, 20, QSizePolicy.Expanding, QSizePolicy.Minimum)

        self.horizontalLayout_22.addItem(self.horizontalSpacer_11)

        self.horizontalLayout_22.setStretch(0, 1)
        self.horizontalLayout_22.setStretch(1, 2)
        self.horizontalLayout_22.setStretch(2, 2)

        self.verticalLayout_11.addLayout(self.horizontalLayout_22)

        self.tabWidget_upload.addTab(self.tab_8, "")
        self.tab_9 = QWidget()
        self.tab_9.setObjectName(u"tab_9")
        self.horizontalLayout_28 = QHBoxLayout(self.tab_9)
        self.horizontalLayout_28.setObjectName(u"horizontalLayout_28")
        self.verticalLayout_14 = QVBoxLayout()
        self.verticalLayout_14.setObjectName(u"verticalLayout_14")
        self.horizontalLayout_23 = QHBoxLayout()
        self.horizontalLayout_23.setObjectName(u"horizontalLayout_23")
        self.label_20 = QLabel(self.tab_9)
        self.label_20.setObjectName(u"label_20")

        self.horizontalLayout_23.addWidget(self.label_20)

        self.cB_choose_diy_layer = QComboBox(self.tab_9)
        self.cB_choose_diy_layer.addItem("")
        self.cB_choose_diy_layer.addItem("")
        self.cB_choose_diy_layer.addItem("")
        self.cB_choose_diy_layer.addItem("")
        self.cB_choose_diy_layer.setObjectName(u"cB_choose_diy_layer")

        self.horizontalLayout_23.addWidget(self.cB_choose_diy_layer)

        self.horizontalSpacer_12 = QSpacerItem(40, 20, QSizePolicy.Expanding, QSizePolicy.Minimum)

        self.horizontalLayout_23.addItem(self.horizontalSpacer_12)

        self.horizontalLayout_23.setStretch(0, 1)
        self.horizontalLayout_23.setStretch(1, 2)
        self.horizontalLayout_23.setStretch(2, 2)

        self.verticalLayout_14.addLayout(self.horizontalLayout_23)

        self.horizontalLayout_24 = QHBoxLayout()
        self.horizontalLayout_24.setObjectName(u"horizontalLayout_24")
        self.label_21 = QLabel(self.tab_9)
        self.label_21.setObjectName(u"label_21")

        self.horizontalLayout_24.addWidget(self.label_21)

        self.iE_diy_layer_params = QLineEdit(self.tab_9)
        self.iE_diy_layer_params.setObjectName(u"iE_diy_layer_params")

        self.horizontalLayout_24.addWidget(self.iE_diy_layer_params)


        self.verticalLayout_14.addLayout(self.horizontalLayout_24)

        self.horizontalLayout_25 = QHBoxLayout()
        self.horizontalLayout_25.setObjectName(u"horizontalLayout_25")
        self.horizontalSpacer_13 = QSpacerItem(40, 20, QSizePolicy.Expanding, QSizePolicy.Minimum)

        self.horizontalLayout_25.addItem(self.horizontalSpacer_13)

        self.pB_add_diy_layer = QPushButton(self.tab_9)
        self.pB_add_diy_layer.setObjectName(u"pB_add_diy_layer")

        self.horizontalLayout_25.addWidget(self.pB_add_diy_layer)

        self.horizontalSpacer_14 = QSpacerItem(40, 20, QSizePolicy.Expanding, QSizePolicy.Minimum)

        self.horizontalLayout_25.addItem(self.horizontalSpacer_14)


        self.verticalLayout_14.addLayout(self.horizontalLayout_25)


        self.horizontalLayout_28.addLayout(self.verticalLayout_14)

        self.horizontalLayout_27 = QHBoxLayout()
        self.horizontalLayout_27.setObjectName(u"horizontalLayout_27")
        self.treeWidget = QTreeWidget(self.tab_9)
        QTreeWidgetItem(self.treeWidget)
        QTreeWidgetItem(self.treeWidget)
        QTreeWidgetItem(self.treeWidget)
        QTreeWidgetItem(self.treeWidget)
        self.treeWidget.setObjectName(u"treeWidget")

        self.horizontalLayout_27.addWidget(self.treeWidget)

        self.verticalLayout_2 = QVBoxLayout()
        self.verticalLayout_2.setObjectName(u"verticalLayout_2")
        self.pushButton_5 = QPushButton(self.tab_9)
        self.pushButton_5.setObjectName(u"pushButton_5")

        self.verticalLayout_2.addWidget(self.pushButton_5)

        self.pushButton_6 = QPushButton(self.tab_9)
        self.pushButton_6.setObjectName(u"pushButton_6")

        self.verticalLayout_2.addWidget(self.pushButton_6)

        self.pushButton_4 = QPushButton(self.tab_9)
        self.pushButton_4.setObjectName(u"pushButton_4")

        self.verticalLayout_2.addWidget(self.pushButton_4)


        self.horizontalLayout_27.addLayout(self.verticalLayout_2)


        self.horizontalLayout_28.addLayout(self.horizontalLayout_27)

        self.tabWidget_upload.addTab(self.tab_9, "")

        self.verticalLayout_5.addWidget(self.tabWidget_upload)

        self.tabWidget.addTab(self.tab_1, "")
        self.tab_2 = QWidget()
        self.tab_2.setObjectName(u"tab_2")
        self.verticalLayout_13 = QVBoxLayout(self.tab_2)
        self.verticalLayout_13.setObjectName(u"verticalLayout_13")
        self.horizontalLayout_26 = QHBoxLayout()
        self.horizontalLayout_26.setObjectName(u"horizontalLayout_26")
        self.cB_fix_method = QComboBox(self.tab_2)
        self.cB_fix_method.addItem("")
        self.cB_fix_method.addItem("")
        self.cB_fix_method.addItem("")
        self.cB_fix_method.setObjectName(u"cB_fix_method")

        self.horizontalLayout_26.addWidget(self.cB_fix_method)

        self.pB_fix = QPushButton(self.tab_2)
        self.pB_fix.setObjectName(u"pB_fix")

        self.horizontalLayout_26.addWidget(self.pB_fix)

        self.pB_download_fixed = QPushButton(self.tab_2)
        self.pB_download_fixed.setObjectName(u"pB_download_fixed")

        self.horizontalLayout_26.addWidget(self.pB_download_fixed)


        self.verticalLayout_13.addLayout(self.horizontalLayout_26)

        self.horizontalLayout_3 = QHBoxLayout()
        self.horizontalLayout_3.setObjectName(u"horizontalLayout_3")
        self.label_15 = QLabel(self.tab_2)
        self.label_15.setObjectName(u"label_15")

        self.horizontalLayout_3.addWidget(self.label_15)

        self.iE_fix_params = QLineEdit(self.tab_2)
        self.iE_fix_params.setObjectName(u"iE_fix_params")

        self.horizontalLayout_3.addWidget(self.iE_fix_params)

        self.horizontalSpacer_17 = QSpacerItem(40, 20, QSizePolicy.Expanding, QSizePolicy.Minimum)

        self.horizontalLayout_3.addItem(self.horizontalSpacer_17)


        self.verticalLayout_13.addLayout(self.horizontalLayout_3)

        self.tabWidget.addTab(self.tab_2, "")
        self.tab_3 = QWidget()
        self.tab_3.setObjectName(u"tab_3")
        self.verticalLayout_9 = QVBoxLayout(self.tab_3)
        self.verticalLayout_9.setObjectName(u"verticalLayout_9")
        self.tabWidget__inject = QTabWidget(self.tab_3)
        self.tabWidget__inject.setObjectName(u"tabWidget__inject")
        self.tab = QWidget()
        self.tab.setObjectName(u"tab")
        self.verticalLayout_6 = QVBoxLayout(self.tab)
        self.verticalLayout_6.setObjectName(u"verticalLayout_6")
        self.horizontalLayout_5 = QHBoxLayout()
        self.horizontalLayout_5.setObjectName(u"horizontalLayout_5")
        self.label_3 = QLabel(self.tab)
        self.label_3.setObjectName(u"label_3")

        self.horizontalLayout_5.addWidget(self.label_3)

        self.sB_layer_1 = QSpinBox(self.tab)
        self.sB_layer_1.setObjectName(u"sB_layer_1")
        self.sB_layer_1.setValue(1)

        self.horizontalLayout_5.addWidget(self.sB_layer_1)


        self.verticalLayout_6.addLayout(self.horizontalLayout_5)

        self.horizontalLayout_7 = QHBoxLayout()
        self.horizontalLayout_7.setObjectName(u"horizontalLayout_7")
        self.label_6 = QLabel(self.tab)
        self.label_6.setObjectName(u"label_6")

        self.horizontalLayout_7.addWidget(self.label_6)

        self.sB_location_1_1 = QSpinBox(self.tab)
        self.sB_location_1_1.setObjectName(u"sB_location_1_1")
        self.sB_location_1_1.setValue(1)

        self.horizontalLayout_7.addWidget(self.sB_location_1_1)

        self.sB_location1_2 = QSpinBox(self.tab)
        self.sB_location1_2.setObjectName(u"sB_location1_2")
        self.sB_location1_2.setValue(1)

        self.horizontalLayout_7.addWidget(self.sB_location1_2)

        self.sB_location_1_3 = QSpinBox(self.tab)
        self.sB_location_1_3.setObjectName(u"sB_location_1_3")
        self.sB_location_1_3.setValue(1)

        self.horizontalLayout_7.addWidget(self.sB_location_1_3)

        self.sB_location_1_4 = QSpinBox(self.tab)
        self.sB_location_1_4.setObjectName(u"sB_location_1_4")
        self.sB_location_1_4.setValue(1)

        self.horizontalLayout_7.addWidget(self.sB_location_1_4)


        self.verticalLayout_6.addLayout(self.horizontalLayout_7)

        self.horizontalLayout_8 = QHBoxLayout()
        self.horizontalLayout_8.setObjectName(u"horizontalLayout_8")
        self.label_7 = QLabel(self.tab)
        self.label_7.setObjectName(u"label_7")

        self.horizontalLayout_8.addWidget(self.label_7)

        self.sB_exp_location = QSpinBox(self.tab)
        self.sB_exp_location.setObjectName(u"sB_exp_location")
        self.sB_exp_location.setValue(32)

        self.horizontalLayout_8.addWidget(self.sB_exp_location)


        self.verticalLayout_6.addLayout(self.horizontalLayout_8)

        self.horizontalLayout_2 = QHBoxLayout()
        self.horizontalLayout_2.setObjectName(u"horizontalLayout_2")
        self.label_22 = QLabel(self.tab)
        self.label_22.setObjectName(u"label_22")

        self.horizontalLayout_2.addWidget(self.label_22)

        self.cB_type_single = QComboBox(self.tab)
        self.cB_type_single.addItem("")
        self.cB_type_single.addItem("")
        self.cB_type_single.addItem("")
        self.cB_type_single.setObjectName(u"cB_type_single")

        self.horizontalLayout_2.addWidget(self.cB_type_single)


        self.verticalLayout_6.addLayout(self.horizontalLayout_2)

        self.tabWidget__inject.addTab(self.tab, "")
        self.tab_5 = QWidget()
        self.tab_5.setObjectName(u"tab_5")
        self.verticalLayout_7 = QVBoxLayout(self.tab_5)
        self.verticalLayout_7.setObjectName(u"verticalLayout_7")
        self.horizontalLayout_9 = QHBoxLayout()
        self.horizontalLayout_9.setObjectName(u"horizontalLayout_9")
        self.label_12 = QLabel(self.tab_5)
        self.label_12.setObjectName(u"label_12")

        self.horizontalLayout_9.addWidget(self.label_12)

        self.sB_layer_2 = QSpinBox(self.tab_5)
        self.sB_layer_2.setObjectName(u"sB_layer_2")
        self.sB_layer_2.setValue(1)

        self.horizontalLayout_9.addWidget(self.sB_layer_2)


        self.verticalLayout_7.addLayout(self.horizontalLayout_9)

        self.horizontalLayout_13 = QHBoxLayout()
        self.horizontalLayout_13.setObjectName(u"horizontalLayout_13")
        self.label_8 = QLabel(self.tab_5)
        self.label_8.setObjectName(u"label_8")

        self.horizontalLayout_13.addWidget(self.label_8)

        self.horizontalSpacer_15 = QSpacerItem(40, 20, QSizePolicy.Expanding, QSizePolicy.Minimum)

        self.horizontalLayout_13.addItem(self.horizontalSpacer_15)

        self.iE_rate = QLineEdit(self.tab_5)
        self.iE_rate.setObjectName(u"iE_rate")

        self.horizontalLayout_13.addWidget(self.iE_rate)


        self.verticalLayout_7.addLayout(self.horizontalLayout_13)

        self.horizontalLayout_14 = QHBoxLayout()
        self.horizontalLayout_14.setObjectName(u"horizontalLayout_14")
        self.label_13 = QLabel(self.tab_5)
        self.label_13.setObjectName(u"label_13")

        self.horizontalLayout_14.addWidget(self.label_13)

        self.cB_type_multiply = QComboBox(self.tab_5)
        self.cB_type_multiply.addItem("")
        self.cB_type_multiply.addItem("")
        self.cB_type_multiply.addItem("")
        self.cB_type_multiply.setObjectName(u"cB_type_multiply")

        self.horizontalLayout_14.addWidget(self.cB_type_multiply)


        self.verticalLayout_7.addLayout(self.horizontalLayout_14)

        self.tabWidget__inject.addTab(self.tab_5, "")
        self.tab_6 = QWidget()
        self.tab_6.setObjectName(u"tab_6")
        self.verticalLayout_8 = QVBoxLayout(self.tab_6)
        self.verticalLayout_8.setObjectName(u"verticalLayout_8")
        self.horizontalLayout_10 = QHBoxLayout()
        self.horizontalLayout_10.setObjectName(u"horizontalLayout_10")
        self.label_9 = QLabel(self.tab_6)
        self.label_9.setObjectName(u"label_9")

        self.horizontalLayout_10.addWidget(self.label_9)

        self.sB_layer_3 = QSpinBox(self.tab_6)
        self.sB_layer_3.setObjectName(u"sB_layer_3")
        self.sB_layer_3.setValue(1)

        self.horizontalLayout_10.addWidget(self.sB_layer_3)


        self.verticalLayout_8.addLayout(self.horizontalLayout_10)

        self.horizontalLayout_11 = QHBoxLayout()
        self.horizontalLayout_11.setObjectName(u"horizontalLayout_11")
        self.label_10 = QLabel(self.tab_6)
        self.label_10.setObjectName(u"label_10")

        self.horizontalLayout_11.addWidget(self.label_10)

        self.sB_location_2_1 = QSpinBox(self.tab_6)
        self.sB_location_2_1.setObjectName(u"sB_location_2_1")
        self.sB_location_2_1.setValue(1)

        self.horizontalLayout_11.addWidget(self.sB_location_2_1)

        self.sB_location_2_2 = QSpinBox(self.tab_6)
        self.sB_location_2_2.setObjectName(u"sB_location_2_2")
        self.sB_location_2_2.setValue(1)

        self.horizontalLayout_11.addWidget(self.sB_location_2_2)

        self.sB_location_2_3 = QSpinBox(self.tab_6)
        self.sB_location_2_3.setObjectName(u"sB_location_2_3")
        self.sB_location_2_3.setValue(1)

        self.horizontalLayout_11.addWidget(self.sB_location_2_3)

        self.sB_location_2_4 = QSpinBox(self.tab_6)
        self.sB_location_2_4.setObjectName(u"sB_location_2_4")
        self.sB_location_2_4.setValue(1)

        self.horizontalLayout_11.addWidget(self.sB_location_2_4)


        self.verticalLayout_8.addLayout(self.horizontalLayout_11)

        self.horizontalLayout_12 = QHBoxLayout()
        self.horizontalLayout_12.setObjectName(u"horizontalLayout_12")
        self.label_11 = QLabel(self.tab_6)
        self.label_11.setObjectName(u"label_11")

        self.horizontalLayout_12.addWidget(self.label_11)

        self.horizontalSpacer_16 = QSpacerItem(40, 20, QSizePolicy.Expanding, QSizePolicy.Minimum)

        self.horizontalLayout_12.addItem(self.horizontalSpacer_16)

        self.iE_target = QLineEdit(self.tab_6)
        self.iE_target.setObjectName(u"iE_target")

        self.horizontalLayout_12.addWidget(self.iE_target)


        self.verticalLayout_8.addLayout(self.horizontalLayout_12)

        self.tabWidget__inject.addTab(self.tab_6, "")

        self.verticalLayout_9.addWidget(self.tabWidget__inject)

        self.horizontalLayout_19 = QHBoxLayout()
        self.horizontalLayout_19.setObjectName(u"horizontalLayout_19")
        self.horizontalSpacer_6 = QSpacerItem(40, 20, QSizePolicy.Expanding, QSizePolicy.Minimum)

        self.horizontalLayout_19.addItem(self.horizontalSpacer_6)

        self.pB_inject_origin = QPushButton(self.tab_3)
        self.pB_inject_origin.setObjectName(u"pB_inject_origin")
        sizePolicy1 = QSizePolicy(QSizePolicy.Minimum, QSizePolicy.Fixed)
        sizePolicy1.setHorizontalStretch(0)
        sizePolicy1.setVerticalStretch(0)
        sizePolicy1.setHeightForWidth(self.pB_inject_origin.sizePolicy().hasHeightForWidth())
        self.pB_inject_origin.setSizePolicy(sizePolicy1)

        self.horizontalLayout_19.addWidget(self.pB_inject_origin)

        self.horizontalSpacer_7 = QSpacerItem(40, 20, QSizePolicy.Expanding, QSizePolicy.Minimum)

        self.horizontalLayout_19.addItem(self.horizontalSpacer_7)


        self.verticalLayout_9.addLayout(self.horizontalLayout_19)

        self.horizontalLayout_20 = QHBoxLayout()
        self.horizontalLayout_20.setObjectName(u"horizontalLayout_20")
        self.horizontalSpacer_8 = QSpacerItem(40, 20, QSizePolicy.Expanding, QSizePolicy.Minimum)

        self.horizontalLayout_20.addItem(self.horizontalSpacer_8)

        self.pB_inject_fixed = QPushButton(self.tab_3)
        self.pB_inject_fixed.setObjectName(u"pB_inject_fixed")

        self.horizontalLayout_20.addWidget(self.pB_inject_fixed)

        self.horizontalSpacer_9 = QSpacerItem(40, 20, QSizePolicy.Expanding, QSizePolicy.Minimum)

        self.horizontalLayout_20.addItem(self.horizontalSpacer_9)


        self.verticalLayout_9.addLayout(self.horizontalLayout_20)

        self.tabWidget.addTab(self.tab_3, "")
        self.tab_10 = QWidget()
        self.tab_10.setObjectName(u"tab_10")
        self.verticalLayout_17 = QVBoxLayout(self.tab_10)
        self.verticalLayout_17.setObjectName(u"verticalLayout_17")
        self.groupBox_5 = QGroupBox(self.tab_10)
        self.groupBox_5.setObjectName(u"groupBox_5")
        self.verticalLayout_16 = QVBoxLayout(self.groupBox_5)
        self.verticalLayout_16.setObjectName(u"verticalLayout_16")
        self.horizontalLayout_32 = QHBoxLayout()
        self.horizontalLayout_32.setObjectName(u"horizontalLayout_32")
        self.label_24 = QLabel(self.groupBox_5)
        self.label_24.setObjectName(u"label_24")

        self.horizontalLayout_32.addWidget(self.label_24)

        self.iE_origin_result_3 = QLineEdit(self.groupBox_5)
        self.iE_origin_result_3.setObjectName(u"iE_origin_result_3")
        self.iE_origin_result_3.setEnabled(True)

        self.horizontalLayout_32.addWidget(self.iE_origin_result_3)

        self.horizontalSpacer_22 = QSpacerItem(40, 20, QSizePolicy.Expanding, QSizePolicy.Minimum)

        self.horizontalLayout_32.addItem(self.horizontalSpacer_22)

        self.horizontalLayout_32.setStretch(0, 1)
        self.horizontalLayout_32.setStretch(1, 3)
        self.horizontalLayout_32.setStretch(2, 1)

        self.verticalLayout_16.addLayout(self.horizontalLayout_32)

        self.horizontalLayout_33 = QHBoxLayout()
        self.horizontalLayout_33.setObjectName(u"horizontalLayout_33")
        self.label_25 = QLabel(self.groupBox_5)
        self.label_25.setObjectName(u"label_25")

        self.horizontalLayout_33.addWidget(self.label_25)

        self.iE_fixed_result_3 = QLineEdit(self.groupBox_5)
        self.iE_fixed_result_3.setObjectName(u"iE_fixed_result_3")
        self.iE_fixed_result_3.setEnabled(True)

        self.horizontalLayout_33.addWidget(self.iE_fixed_result_3)

        self.horizontalSpacer_23 = QSpacerItem(40, 20, QSizePolicy.Expanding, QSizePolicy.Minimum)

        self.horizontalLayout_33.addItem(self.horizontalSpacer_23)

        self.horizontalLayout_33.setStretch(0, 1)
        self.horizontalLayout_33.setStretch(1, 3)
        self.horizontalLayout_33.setStretch(2, 1)

        self.verticalLayout_16.addLayout(self.horizontalLayout_33)


        self.verticalLayout_17.addWidget(self.groupBox_5)

        self.groupBox_4 = QGroupBox(self.tab_10)
        self.groupBox_4.setObjectName(u"groupBox_4")
        self.verticalLayout_15 = QVBoxLayout(self.groupBox_4)
        self.verticalLayout_15.setObjectName(u"verticalLayout_15")
        self.horizontalLayout_30 = QHBoxLayout()
        self.horizontalLayout_30.setObjectName(u"horizontalLayout_30")
        self.label_19 = QLabel(self.groupBox_4)
        self.label_19.setObjectName(u"label_19")

        self.horizontalLayout_30.addWidget(self.label_19)

        self.iE_origin_result_2 = QLineEdit(self.groupBox_4)
        self.iE_origin_result_2.setObjectName(u"iE_origin_result_2")
        self.iE_origin_result_2.setEnabled(True)

        self.horizontalLayout_30.addWidget(self.iE_origin_result_2)

        self.horizontalSpacer_20 = QSpacerItem(40, 20, QSizePolicy.Expanding, QSizePolicy.Minimum)

        self.horizontalLayout_30.addItem(self.horizontalSpacer_20)

        self.horizontalLayout_30.setStretch(0, 1)
        self.horizontalLayout_30.setStretch(1, 3)
        self.horizontalLayout_30.setStretch(2, 1)

        self.verticalLayout_15.addLayout(self.horizontalLayout_30)


        self.verticalLayout_17.addWidget(self.groupBox_4)

        self.horizontalLayout_29 = QHBoxLayout()
        self.horizontalLayout_29.setObjectName(u"horizontalLayout_29")
        self.horizontalSpacer_18 = QSpacerItem(40, 20, QSizePolicy.Expanding, QSizePolicy.Minimum)

        self.horizontalLayout_29.addItem(self.horizontalSpacer_18)

        self.pushButton_7 = QPushButton(self.tab_10)
        self.pushButton_7.setObjectName(u"pushButton_7")

        self.horizontalLayout_29.addWidget(self.pushButton_7)

        self.horizontalSpacer_19 = QSpacerItem(40, 20, QSizePolicy.Expanding, QSizePolicy.Minimum)

        self.horizontalLayout_29.addItem(self.horizontalSpacer_19)


        self.verticalLayout_17.addLayout(self.horizontalLayout_29)

        self.tabWidget.addTab(self.tab_10, "")
        self.tab_4 = QWidget()
        self.tab_4.setObjectName(u"tab_4")
        self.verticalLayout_12 = QVBoxLayout(self.tab_4)
        self.verticalLayout_12.setObjectName(u"verticalLayout_12")
        self.groupBox_3 = QGroupBox(self.tab_4)
        self.groupBox_3.setObjectName(u"groupBox_3")
        self.verticalLayout_10 = QVBoxLayout(self.groupBox_3)
        self.verticalLayout_10.setObjectName(u"verticalLayout_10")
        self.horizontalLayout_17 = QHBoxLayout()
        self.horizontalLayout_17.setObjectName(u"horizontalLayout_17")
        self.label_14 = QLabel(self.groupBox_3)
        self.label_14.setObjectName(u"label_14")

        self.horizontalLayout_17.addWidget(self.label_14)

        self.iE_origin_result = QLineEdit(self.groupBox_3)
        self.iE_origin_result.setObjectName(u"iE_origin_result")
        self.iE_origin_result.setEnabled(True)

        self.horizontalLayout_17.addWidget(self.iE_origin_result)

        self.pB_download_origin_inject = QPushButton(self.groupBox_3)
        self.pB_download_origin_inject.setObjectName(u"pB_download_origin_inject")

        self.horizontalLayout_17.addWidget(self.pB_download_origin_inject)

        self.horizontalSpacer_2 = QSpacerItem(40, 20, QSizePolicy.Expanding, QSizePolicy.Minimum)

        self.horizontalLayout_17.addItem(self.horizontalSpacer_2)

        self.horizontalLayout_17.setStretch(0, 1)
        self.horizontalLayout_17.setStretch(1, 3)
        self.horizontalLayout_17.setStretch(3, 1)

        self.verticalLayout_10.addLayout(self.horizontalLayout_17)

        self.horizontalLayout_16 = QHBoxLayout()
        self.horizontalLayout_16.setObjectName(u"horizontalLayout_16")
        self.label_16 = QLabel(self.groupBox_3)
        self.label_16.setObjectName(u"label_16")

        self.horizontalLayout_16.addWidget(self.label_16)

        self.iE_fixed_result = QLineEdit(self.groupBox_3)
        self.iE_fixed_result.setObjectName(u"iE_fixed_result")
        self.iE_fixed_result.setEnabled(True)

        self.horizontalLayout_16.addWidget(self.iE_fixed_result)

        self.pB_dowload_fixed_inject = QPushButton(self.groupBox_3)
        self.pB_dowload_fixed_inject.setObjectName(u"pB_dowload_fixed_inject")

        self.horizontalLayout_16.addWidget(self.pB_dowload_fixed_inject)

        self.horizontalSpacer_3 = QSpacerItem(40, 20, QSizePolicy.Expanding, QSizePolicy.Minimum)

        self.horizontalLayout_16.addItem(self.horizontalSpacer_3)

        self.horizontalLayout_16.setStretch(0, 1)
        self.horizontalLayout_16.setStretch(1, 3)
        self.horizontalLayout_16.setStretch(3, 1)

        self.verticalLayout_10.addLayout(self.horizontalLayout_16)


        self.verticalLayout_12.addWidget(self.groupBox_3)

        self.tabWidget.addTab(self.tab_4, "")
        MainWindow.setCentralWidget(self.centralwidget)
        self.menubar = QMenuBar(MainWindow)
        self.menubar.setObjectName(u"menubar")
        self.menubar.setGeometry(QRect(0, 0, 951, 26))
        MainWindow.setMenuBar(self.menubar)

        self.retranslateUi(MainWindow)

        self.tabWidget.setCurrentIndex(0)
        self.tabWidget_upload.setCurrentIndex(0)
        self.tabWidget__inject.setCurrentIndex(0)


        QMetaObject.connectSlotsByName(MainWindow)
    # setupUi

    def retranslateUi(self, MainWindow):
        MainWindow.setWindowTitle(QCoreApplication.translate("MainWindow", u"MainWindow", None))
        self.pushButton_2.setText("")
        self.pushButton.setText("")
        self.pushButton_3.setText("")
        self.label.setText(QCoreApplication.translate("MainWindow", u"\u795e\u7ecf\u7f51\u7edc\u9519\u8bef\u4eff\u771f\u4e0e\u5bb9\u9519\u589e\u5f3a\u7cfb\u7edf", None))
        self.groupBox_2.setTitle(QCoreApplication.translate("MainWindow", u"\u6a21\u578b\u7ed3\u6784", None))
        self.label12.setText(QCoreApplication.translate("MainWindow", u"\u539f\u6a21\u578b", None))
        self.tB_origin_model.setHtml(QCoreApplication.translate("MainWindow", u"<!DOCTYPE HTML PUBLIC \"-//W3C//DTD HTML 4.0//EN\" \"http://www.w3.org/TR/REC-html40/strict.dtd\">\n"
"<html><head><meta name=\"qrichtext\" content=\"1\" /><style type=\"text/css\">\n"
"p, li { white-space: pre-wrap; }\n"
"</style></head><body style=\" font-family:'SimSun'; font-size:9pt; font-weight:400; font-style:normal;\">\n"
"<p style=\"-qt-paragraph-type:empty; margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;\"><br /></p></body></html>", None))
        self.label_2.setText(QCoreApplication.translate("MainWindow", u"\u589e\u5f3a\u6a21\u578b", None))
        self.label_4.setText(QCoreApplication.translate("MainWindow", u"\u8bf7\u9009\u62e9\u6a21\u578b\uff1a", None))
        self.pB_upload_model.setText(QCoreApplication.translate("MainWindow", u"\u9009\u62e9\u6587\u4ef6", None))
        self.iE_model_uri.setText(QCoreApplication.translate("MainWindow", u"/home/madao/pyqt/lene5.h5", None))
        self.label_5.setText(QCoreApplication.translate("MainWindow", u"\u9009\u62e9\u6570\u636e\u96c6\uff1a", None))
        self.pB_upload_data.setText(QCoreApplication.translate("MainWindow", u"\u4e0a\u4f20\u6587\u4ef6", None))
        self.iE_data_uri.setText(QCoreApplication.translate("MainWindow", u"/home/madao/pyqt/data/", None))
        self.tabWidget_upload.setTabText(self.tabWidget_upload.indexOf(self.tab_7), QCoreApplication.translate("MainWindow", u"\u52a0\u8f7d\u6a21\u578b", None))
        self.label_17.setText(QCoreApplication.translate("MainWindow", u"\u8bf7\u9009\u62e9\u6a21\u578b\uff1a", None))
        self.cB_choose_model.setItemText(0, QCoreApplication.translate("MainWindow", u"Lenet5", None))
        self.cB_choose_model.setItemText(1, QCoreApplication.translate("MainWindow", u"Resnet", None))
        self.cB_choose_model.setItemText(2, QCoreApplication.translate("MainWindow", u"Googlenet", None))

        self.label_18.setText(QCoreApplication.translate("MainWindow", u"\u9009\u62e9\u6570\u636e\u96c6\uff1a", None))
        self.cB_choose_data.setItemText(0, QCoreApplication.translate("MainWindow", u"coco", None))
        self.cB_choose_data.setItemText(1, QCoreApplication.translate("MainWindow", u"minst", None))

        self.cB_choose_data.setCurrentText(QCoreApplication.translate("MainWindow", u"coco", None))
        self.tabWidget_upload.setTabText(self.tabWidget_upload.indexOf(self.tab_8), QCoreApplication.translate("MainWindow", u"\u6807\u51c6\u6a21\u578b\u5bfc\u5165", None))
        self.label_20.setText(QCoreApplication.translate("MainWindow", u"\u81ea\u5b9a\u4e49\u5c42\uff1a", None))
        self.cB_choose_diy_layer.setItemText(0, QCoreApplication.translate("MainWindow", u"\u5377\u79ef\u5c42", None))
        self.cB_choose_diy_layer.setItemText(1, QCoreApplication.translate("MainWindow", u"\u6c60\u5316\u5c42", None))
        self.cB_choose_diy_layer.setItemText(2, QCoreApplication.translate("MainWindow", u"\u5168\u8fde\u63a5\u5c42", None))
        self.cB_choose_diy_layer.setItemText(3, QCoreApplication.translate("MainWindow", u"\u6fc0\u6d3b\u5c42", None))

        self.label_21.setText(QCoreApplication.translate("MainWindow", u"\u8d85\u53c2\u6570\uff1a", None))
        self.iE_diy_layer_params.setText(QCoreApplication.translate("MainWindow", u"64;3,3;1,1;1", None))
        self.pB_add_diy_layer.setText(QCoreApplication.translate("MainWindow", u"\u52a0\u5165", None))
        ___qtreewidgetitem = self.treeWidget.headerItem()
        ___qtreewidgetitem.setText(0, QCoreApplication.translate("MainWindow", u"model", None));

        __sortingEnabled = self.treeWidget.isSortingEnabled()
        self.treeWidget.setSortingEnabled(False)
        ___qtreewidgetitem1 = self.treeWidget.topLevelItem(0)
        ___qtreewidgetitem1.setText(0, QCoreApplication.translate("MainWindow", u"conv2d-1(64;3,3;1,1;1)", None));
        ___qtreewidgetitem2 = self.treeWidget.topLevelItem(1)
        ___qtreewidgetitem2.setText(0, QCoreApplication.translate("MainWindow", u"relu", None));
        ___qtreewidgetitem3 = self.treeWidget.topLevelItem(2)
        ___qtreewidgetitem3.setText(0, QCoreApplication.translate("MainWindow", u"conv2d-2(64;3,3;1,1;1)", None));
        ___qtreewidgetitem4 = self.treeWidget.topLevelItem(3)
        ___qtreewidgetitem4.setText(0, QCoreApplication.translate("MainWindow", u"relu", None));
        self.treeWidget.setSortingEnabled(__sortingEnabled)

        self.pushButton_5.setText(QCoreApplication.translate("MainWindow", u"\u5220\u9664", None))
        self.pushButton_6.setText(QCoreApplication.translate("MainWindow", u"\u751f\u6210", None))
        self.pushButton_4.setText(QCoreApplication.translate("MainWindow", u"\u53e6\u5b58\u4e3a", None))
        self.tabWidget_upload.setTabText(self.tabWidget_upload.indexOf(self.tab_9), QCoreApplication.translate("MainWindow", u"\u81ea\u5b9a\u4e49\u6a21\u578b\u5bfc\u5165", None))
        self.tabWidget.setTabText(self.tabWidget.indexOf(self.tab_1), QCoreApplication.translate("MainWindow", u"\u6570\u636e\u4e0a\u4f20", None))
        self.cB_fix_method.setItemText(0, QCoreApplication.translate("MainWindow", u"BRelu", None))
        self.cB_fix_method.setItemText(1, QCoreApplication.translate("MainWindow", u"ChRelu", None))
        self.cB_fix_method.setItemText(2, QCoreApplication.translate("MainWindow", u"\u5bb9\u9519\u65b9\u6848\u4e09", None))

        self.pB_fix.setText(QCoreApplication.translate("MainWindow", u"\u589e\u5f3a", None))
        self.pB_download_fixed.setText(QCoreApplication.translate("MainWindow", u"\u4e0b\u8f7d\u6a21\u578b", None))
        self.label_15.setText(QCoreApplication.translate("MainWindow", u"\u8d85\u53c2\u6570\uff1a", None))
        self.tabWidget.setTabText(self.tabWidget.indexOf(self.tab_2), QCoreApplication.translate("MainWindow", u"\u5bb9\u9519\u65b9\u6cd5", None))
        self.label_3.setText(QCoreApplication.translate("MainWindow", u"\u5c42\u6570\uff1a", None))
        self.label_6.setText(QCoreApplication.translate("MainWindow", u"\u6743\u91cd\u5750\u6807\uff1a", None))
        self.label_7.setText(QCoreApplication.translate("MainWindow", u"\u6307\u6570\u7ffb\u8f6c\u4f4d\u7f6e\uff1a", None))
        self.label_22.setText(QCoreApplication.translate("MainWindow", u"\u7c7b\u578b\uff1a", None))
        self.cB_type_single.setItemText(0, QCoreApplication.translate("MainWindow", u"\u7f6e1", None))
        self.cB_type_single.setItemText(1, QCoreApplication.translate("MainWindow", u"\u7f6e0", None))
        self.cB_type_single.setItemText(2, QCoreApplication.translate("MainWindow", u"\u7ffb\u8f6c", None))

        self.tabWidget__inject.setTabText(self.tabWidget__inject.indexOf(self.tab), QCoreApplication.translate("MainWindow", u"\u5355\u6bd4\u7279\u9519\u8bef\u6ce8\u5165", None))
        self.label_12.setText(QCoreApplication.translate("MainWindow", u"\u5c42\u6570\uff1a", None))
        self.label_8.setText(QCoreApplication.translate("MainWindow", u"\u9519\u8bef\u7387\uff1a", None))
        self.label_13.setText(QCoreApplication.translate("MainWindow", u"\u7c7b\u578b\uff1a", None))
        self.cB_type_multiply.setItemText(0, QCoreApplication.translate("MainWindow", u"\u7f6e1", None))
        self.cB_type_multiply.setItemText(1, QCoreApplication.translate("MainWindow", u"\u7f6e0", None))
        self.cB_type_multiply.setItemText(2, QCoreApplication.translate("MainWindow", u"\u7ffb\u8f6c", None))

        self.tabWidget__inject.setTabText(self.tabWidget__inject.indexOf(self.tab_5), QCoreApplication.translate("MainWindow", u"\u591a\u6bd4\u7279\u9519\u8bef\u6ce8\u5165", None))
        self.label_9.setText(QCoreApplication.translate("MainWindow", u"\u5c42\u6570\uff1a", None))
        self.label_10.setText(QCoreApplication.translate("MainWindow", u"\u6743\u91cd\u5750\u6807\uff1a", None))
        self.label_11.setText(QCoreApplication.translate("MainWindow", u"\u6307\u5b9a\u6570\uff1a", None))
        self.tabWidget__inject.setTabText(self.tabWidget__inject.indexOf(self.tab_6), QCoreApplication.translate("MainWindow", u"\u6307\u5b9a\u6570\u7ffb\u8f6c", None))
        self.pB_inject_origin.setText(QCoreApplication.translate("MainWindow", u"\u6ce8\u5165\u539f\u6a21\u578b", None))
        self.pB_inject_fixed.setText(QCoreApplication.translate("MainWindow", u"\u6ce8\u5165\u5bb9\u9519\u540e\u7684\u6a21\u578b", None))
        self.tabWidget.setTabText(self.tabWidget.indexOf(self.tab_3), QCoreApplication.translate("MainWindow", u"\u9519\u8bef\u6ce8\u5165", None))
        self.groupBox_5.setTitle(QCoreApplication.translate("MainWindow", u"\u989d\u5916\u5f00\u9500", None))
        self.label_24.setText(QCoreApplication.translate("MainWindow", u"\u989d\u5916\u65f6\u95f4\u5f00\u9500\u7ed3\u679c", None))
        self.iE_origin_result_3.setText(QCoreApplication.translate("MainWindow", u"7.63%", None))
        self.label_25.setText(QCoreApplication.translate("MainWindow", u"\u989d\u5916\u7a7a\u95f4\u5f00\u9500\u7ed3\u679c", None))
        self.iE_fixed_result_3.setText(QCoreApplication.translate("MainWindow", u"5.14%", None))
        self.groupBox_4.setTitle(QCoreApplication.translate("MainWindow", u"\u7cbe\u5ea6\u635f\u5931", None))
        self.label_19.setText(QCoreApplication.translate("MainWindow", u"\u5bb9\u9519\u65b9\u6cd5\u7cbe\u5ea6\u635f\u5931", None))
        self.iE_origin_result_2.setText(QCoreApplication.translate("MainWindow", u"0.28%", None))
        self.pushButton_7.setText(QCoreApplication.translate("MainWindow", u"\u4e00\u952e\u5206\u6790", None))
        self.tabWidget.setTabText(self.tabWidget.indexOf(self.tab_10), QCoreApplication.translate("MainWindow", u"\u6570\u636e\u5206\u6790", None))
        self.groupBox_3.setTitle(QCoreApplication.translate("MainWindow", u"\u9884\u6d4b\u7ed3\u679c", None))
        self.label_14.setText(QCoreApplication.translate("MainWindow", u"\u539f\u6a21\u578b\u7cbe\u5ea6\uff1a", None))
        self.iE_origin_result.setText(QCoreApplication.translate("MainWindow", u"16.98%", None))
        self.pB_download_origin_inject.setText(QCoreApplication.translate("MainWindow", u"\u9884\u6d4b", None))
        self.label_16.setText(QCoreApplication.translate("MainWindow", u"\u589e\u5f3a\u6a21\u578b\u7cbe\u5ea6\uff1a", None))
        self.iE_fixed_result.setText(QCoreApplication.translate("MainWindow", u"70.76%", None))
        self.pB_dowload_fixed_inject.setText(QCoreApplication.translate("MainWindow", u"\u9884\u6d4b", None))
        self.tabWidget.setTabText(self.tabWidget.indexOf(self.tab_4), QCoreApplication.translate("MainWindow", u"\u7ed3\u679c\u67e5\u770b", None))
    # retranslateUi

