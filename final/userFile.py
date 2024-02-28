import torch
import torch.nn as nn
from torchvision import models
import torchsummary
from software1 import fixModel
from torchvision import models
import warnings

warnings.filterwarnings('ignore')
class LeNet(nn.Module):
    def __init__(self):
        super(LeNet, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=6, kernel_size=5, stride=1)
        self.relu1 = nn.ReLU()
        self.mp1 = nn.MaxPool2d(kernel_size=2)
        self.conv2 = nn.Conv2d(in_channels=6, out_channels=16, kernel_size=5, stride=1)
        self.relu2 = nn.ReLU()
        self.mp2 = nn.MaxPool2d(kernel_size=2)
        self.flatten=nn.Flatten()
        self.fc1 = nn.Linear(in_features=256, out_features=120)
        self.relu3 = nn.ReLU()
        self.fc2 = nn.Linear(in_features=120, out_features=84)
        self.relu4 = nn.ReLU()
        self.fc3 = nn.Linear(in_features=84, out_features=10)

    def forward(self, x):
        x = self.mp1(self.relu1(self.conv1(x)))
        x = self.conv2(x)
        x = self.mp2(self.relu2(x))
        x = self.flatten(x)
        x = self.relu3(self.fc1(x))
        x = self.relu4(self.fc2(x))
        x = self.fc3(x)
        return x


class ComplexModel(nn.Module):
    def __init__(self):
        super(ComplexModel, self).__init__()
        self.resnet = models.resnet50(pretrained=True)  # 使用预训练的ResNet50作为基础模型
        num_ftrs = self.resnet.fc.in_features
        # 冻结ResNet的参数
        for param in self.resnet.parameters():
            param.requires_grad = False
        # 替换ResNet的全连接层
        self.resnet.fc = nn.Sequential(
            nn.Linear(num_ftrs, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 10)  # 假设输出类别数为10
        )

    def forward(self, x):
        return self.resnet(x)


class CustomResNet18(nn.Module):
    def __init__(self, num_classes):
        super(CustomResNet18, self).__init__()
        self.resnet18 = models.resnet18(pretrained=True)
        in_features = self.resnet18.fc.in_features
        self.resnet18.fc = nn.Linear(in_features, num_classes)

    def forward(self, x):
        return self.resnet18(x)


if __name__ == '__main__':
    model=ComplexModel()

    # model = torch.load(r"D:\容错\model\vgg16_model.pth",map_location='cpu')
    # torch.save(model,'/home/letian.chen/data/resnet_full.pth')

    # fixModel(model, (224, 224, 3))
    if type(model) is models.vgg.VGG:
        fixModel(model,(224,224,3))
    if type(model) is LeNet:
        fixModel(model, (28, 28, 1))
    if type(model) is ComplexModel:
        fixModel(model, (224,224,3))