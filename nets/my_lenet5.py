import torch.nn as nn
import torch
import torch.nn.functional as F

class LeNet5(nn.Module):
    def __init__(self, num_classes):
        """
        num_classes: 分类的数量
        grayscale：是否为灰度图
        """
        super(LeNet5, self).__init__()
        self.num_classes = num_classes

        # 卷积神经网络
        self.features = nn.Sequential(
            nn.Conv2d(1, 6, kernel_size=5),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(6, 16, kernel_size=5),
            nn.MaxPool2d(kernel_size=2)
        )
        # 分类器
        self.classifier = nn.Sequential(
            nn.Linear(16*4*4, 120),  # 这里把第三个卷积当作是全连接层了
            nn.Linear(120, 84),
            nn.Linear(84, num_classes)
        )

    def forward(self, x):
        x = self.features(x) # 输出 16*5*5 特征图
        x = torch.flatten(x, 1) # 展平 （1， 16*4*4）
        logits = self.classifier(x) # 输出 10
        probas = F.softmax(logits, dim=1)
        return probas