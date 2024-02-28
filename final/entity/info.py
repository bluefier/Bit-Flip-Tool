from PyQt5.QtCore import QThread
from PySide2.QtWidgets import QDialog, QProgressDialog
from PySide2.QtUiTools import QUiLoader
from PyQt5.QtWidgets import QApplication, QWidget, QVBoxLayout, QProgressBar
from PyQt5.QtCore import QTimer
# from PyQt5.QtWidgets import QApplication, QProgressBar, QTimer, QVBoxLayout, QWidget, QThread

#创建子窗口类
class infoWindow(QDialog):
    def __init__(self,msg:str):
        super(infoWindow, self).__init__()  # type: ignore
        #引入子窗口类
        self.ui = QUiLoader().load('ui/info.ui')
        self.ui.pushButton.clicked.connect(self.return_main)
        self.ui.textBrowser.setPlainText(msg)
    def close(self):
        self.ui.close()   # type: ignore
    def return_main(self):
        self.ui.close()

class Info:
    def __init__(self) -> None:
        self.model = None
        self.fixed_model = None
        self.fixed_inject_model = None
        self.origin_inject_model = None
        # self.normal_inject_model = None
        # self.net = None
        self.convs = [] 
        self.fixed_convs = [] 
        
        self.layer_num = 0 # 单指有参数的卷积层。




class PropThread(QThread):
    def __init__(self):
        super().__init__()

    def run(self):
        self.proBar = ProgressBar()
        self.proBar.show()

        # 创建一个定时器，模拟进度的增长
        timer = QTimer(self)
        timer.timeout.connect(self.proBar.updateProgress)
        timer.start(100)  # 每100毫秒更新一次进度

        self.exec_()  # 运行当前线程的事件循环

class ProgressBar(QWidget):
    def __init__(self):
        super().__init__()

        layout = QVBoxLayout(self)

        self.progress_bar = QProgressBar(self)
        layout.addWidget(self.progress_bar)

    def updateProgress(self):
        # 更新进度条的值
        current_value = self.progress_bar.value()
        new_value = (current_value + 1) % 101  # 循环从0到100
        self.progress_bar.setValue(new_value)

        if new_value == 100:  # 例如，当进度达到80时自动关闭
            self.close()

if __name__ == "__main__":
    app = QApplication([])


    # 创建并启动线程
    p = ProgressBar()
    p.show()

    # 执行主事件循环
    app.exec_()

