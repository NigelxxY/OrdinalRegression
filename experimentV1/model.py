import torch
import torch.nn as nn
import torch.nn.functional as F


class Residual(nn.Module):
    def __init__(self):
        super(Residual, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=1,out_channels=8,kernel_size=5,stride=1,padding=2)
        self.conv2 = nn.Conv2d(in_channels=8,out_channels=1,kernel_size=5,stride=1,padding=2)
        self.relu = nn.ReLU()
        self.pooling = nn.MaxPool2d(kernel_size=3,stride=1,padding=1)

    def forward(self,x):
        residual = x
        x = self.conv1(x)
        x = self.relu(x)
        x = self.pooling(x)
        x = self.conv2(x)
        x = self.relu(x)
        x = self.pooling(x)
        x = x + residual
        out = self.relu(x)
        # print("conv1 param: \n",self.conv1.weight)
        # print("conv2 param: \n",self.conv2.weight)
        return out


class FcnForRegression(nn.Module):

    def __init__(self):
        super(FcnForRegression, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=1,out_channels=4,kernel_size=3,stride=1)
        self.conv2 = nn.Conv2d(in_channels=4,out_channels=1,kernel_size=3,stride=1)
        self.relu = nn.ReLU()
        self.pooling = nn.MaxPool2d(kernel_size=3,stride=1)

    def forward(self,x):
        x = self.conv1(x)
        x = self.relu(x)
        x = self.pooling(x)
        x = self.conv2(x)
        x = self.relu(x)
        x = self.pooling(x)
        return x.view(x.shape[0],-1).mean(1).view(-1,1)