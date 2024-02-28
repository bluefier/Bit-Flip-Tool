import torch.nn as nn
import torch.nn.functional as F


class ResidualBlock(nn.Module):
    def __init__(self,channels):
        super(ResidualBlock,self).__init__()
        self.channels = channels
        self.conv1 = nn.Conv2d(channels,channels,kernel_size=3,padding=1)
        self.conv2  = nn.Conv2d(channels,channels,kernel_size=3,padding =1)
        self.relu1 = nn.ReLU()
    def forward(self,x):
        y = self.relu1(self.conv1(x))
        y = self.conv2(y)
        return self.relu1(x+y)

class Net(nn.Module):
    def __init__(self) -> None:
        super(Net,self).__init__()
        self.conv1 = nn.Conv2d(1,16,kernel_size=5)
        self.conv2 = nn.Conv2d(16,32,kernel_size=5)
        self.mp = nn.MaxPool2d(2)
        self.m_relu = nn.ReLU()
        
        self.fc = nn.Linear(512,10)

        self.rblock1 = ResidualBlock(16)
        self.rblock2 = ResidualBlock(32)

        
    def forward(self,x):
        in_size = x.size(0)
        x = self.mp(self.m_relu(self.conv1(x)))
        x = self.rblock1(x)
        x = self.mp(self.m_relu(self.conv2(x)))
        x = self.rblock2(x)
        x = x.view(in_size,-1)
        x = self.fc(x)
        return x


