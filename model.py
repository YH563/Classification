import torch
import torch.nn as nn
import torch.nn.init as init
from torch.nn import functional as F


# 通道注意力模块
class ChannelAttention(nn.Module):
    def __init__(self, in_channels, ratio=16):
        super(ChannelAttention, self).__init__()
        # 全局平均池化和最大池化
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.fc = nn.Sequential(
            nn.Conv2d(in_channels, in_channels//ratio, 1),
            nn.ReLU(),
            nn.Conv2d(in_channels//ratio, in_channels, 1)
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc(self.avg_pool(x))
        max_out = self.fc(self.max_pool(x))
        out = avg_out + max_out
        return self.sigmoid(out)


# 空间注意力模块
class SpatialAttention(nn.Module):
    def __init__(self):
        super(SpatialAttention, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=2, out_channels=1, kernel_size=7, stride=1, padding=3)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # 通道平均和最大池化
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv1(x)
        return self.sigmoid(x)


# 通道空间双重增强的注意力模块
class ConvolutionalBlockAttentionModule(nn.Module):
    def __init__(self, in_channels, ratio=16):
        super(ConvolutionalBlockAttentionModule, self).__init__()
        self.ca = ChannelAttention(in_channels, ratio)
        self.sa = SpatialAttention()

    def forward(self, x):
        out = self.ca(x) * x
        out = self.sa(out) * out
        return out


# 定义残差块，融合了注意力模块以优化分类效果
class ResidualBlock(nn.Module):
    def __init__(self, in_channels):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(in_channels, in_channels, kernel_size=1, stride=1)
        self.cbam = ConvolutionalBlockAttentionModule(in_channels)
        self.bn = nn.BatchNorm2d(in_channels, track_running_stats=True)
        # 对卷积层进行凯明初始化
        init.kaiming_normal_(self.conv1.weight, mode='fan_out', nonlinearity='relu')
        init.kaiming_normal_(self.conv2.weight, mode='fan_out', nonlinearity='relu')

    def forward(self, x):
        conv2_out = self.conv2(x)
        x = self.conv1(x)
        x = self.bn(x)
        x = F.relu(x)
        x = self.conv1(x)
        # x = self.cbam(x)
        x = self.bn(x)
        return F.relu(x + conv2_out)


# 分类模型
class ClassificationModel(nn.Module):
    def __init__(self):
        super(ClassificationModel, self).__init__()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.fc1 = nn.Linear(512 * 8 * 8, 512)
        self.fc2 = nn.Linear(512, 4)
        self.convList = nn.ModuleList()
        self.resList = nn.ModuleList()
        self.bnList = nn.ModuleList()
        for i in range(4):
            out_channels = 2**(6 + i)
            in_channels = out_channels//2 if i != 0 else 3
            if i == 0:
                conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=2, padding=1)
            else:
                conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1)
            self.convList.append(conv)
            self.bnList.append(nn.BatchNorm2d(out_channels))
            self.resList.append(ResidualBlock(out_channels))

            # 对卷积层进行凯明初始化
            init.kaiming_normal_(conv.weight, mode='fan_out', nonlinearity='relu')

    def forward(self, x):
        for conv, bn, res in zip(self.convList, self.bnList, self.resList):
            x = conv(x)
            x = F.relu(bn(x))
            x = self.pool(x)
            x = res(x)
        x = self.pool(x)
        x = nn.Flatten()(x)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x
