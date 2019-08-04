import torch
import torch.nn as nn
import torch.nn.functional as F
import os

class BasicConv(nn.Module):

    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, dilation=1, groups=1, relu=True, bn=True):
        super(BasicConv, self).__init__()
        self.out_channels = out_planes
        if bn:
            self.conv = nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation, groups=groups, bias=False)
            self.bn = nn.BatchNorm2d(out_planes, eps=1e-5, momentum=0.01, affine=True)
            self.relu = nn.ReLU(inplace=True) if relu else None
        else:
            self.conv = nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation, groups=groups, bias=True)
            self.bn = None
            self.relu = nn.ReLU(inplace=True) if relu else None

    def forward(self, x):
        x = self.conv(x)
        if self.bn is not None:
            x = self.bn(x)
        if self.relu is not None:
            x = self.relu(x)
        return x


class BasicRFB(nn.Module):

    def __init__(self, in_planes, out_planes, stride=1, scale = 0.1, map_reduce=8, vision=1, groups=1):
        super(BasicRFB, self).__init__()
        self.scale = scale
        self.out_channels = out_planes
        inter_planes = in_planes // map_reduce

        self.branch0 = nn.Sequential(
                BasicConv(in_planes, inter_planes, kernel_size=1, stride=1, groups=groups, relu=False),
                BasicConv(inter_planes, 2 * inter_planes, kernel_size=(3, 3), stride=stride, padding=(1, 1), groups=groups),
                BasicConv(2 * inter_planes, 2 * inter_planes, kernel_size=3, stride=1, padding=vision+1, dilation=vision+1, relu=False, groups=groups)
                )
        self.branch1 = nn.Sequential(
                BasicConv(in_planes, inter_planes, kernel_size=1, stride=1, groups=groups, relu=False),
                BasicConv(inter_planes, 2*inter_planes, kernel_size=(3,3), stride=stride, padding=(1,1), groups=groups),
                BasicConv(2*inter_planes, 2*inter_planes, kernel_size=3, stride=1, padding=vision + 2, dilation=vision + 2, relu=False, groups=groups)
                )
        self.branch2 = nn.Sequential(
                BasicConv(in_planes, inter_planes, kernel_size=1, stride=1, groups=groups, relu=False),
                BasicConv(inter_planes, (inter_planes//2)*3, kernel_size=3, stride=1, padding=1, groups=groups),
                BasicConv((inter_planes//2)*3, 2*inter_planes, kernel_size=3, stride=stride, padding=1, groups=groups),
                BasicConv(2*inter_planes, 2*inter_planes, kernel_size=3, stride=1, padding=vision + 4, dilation=vision + 4, relu=False, groups=groups)
                )

        self.ConvLinear = BasicConv(6*inter_planes, out_planes, kernel_size=1, stride=1, relu=False)
        self.shortcut = BasicConv(in_planes, out_planes, kernel_size=1, stride=stride, relu=False)
        self.relu = nn.ReLU(inplace=False)

    def forward(self,x):
        x0 = self.branch0(x)
        x1 = self.branch1(x)
        x2 = self.branch2(x)

        out = torch.cat((x0,x1,x2),1)
        out = self.ConvLinear(out)
        short = self.shortcut(x)
        out = out*self.scale + short
        out = self.relu(out)

        return out

class Backbone(nn.Module):
    def __init__(self, bn=True):
        super(Backbone, self).__init__()

        self.conv1_1 = BasicConv(3,  32, kernel_size=3, padding=1, bn=bn)
        self.conv1_2 = BasicConv(32, 32, kernel_size=3, padding=1, bn=bn)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)  # 2

        self.conv2_1 = BasicConv(32, 64, kernel_size=3, padding=1, bn=bn)
        self.conv2_2 = BasicConv(64, 64, kernel_size=3, padding=1, bn=bn)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2) # 4

        self.conv3_1 = BasicConv(64, 128, kernel_size=1, bn=bn)
        self.conv3_2 = BasicConv(128, 128, kernel_size=3, padding=1, bn=bn)
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=bn)  # 8

        self.conv4_1 = BasicConv(128, 256, kernel_size=1, bn=bn)
        self.conv4_2 = BasicConv(256, 256, kernel_size=3, padding=1, bn=bn)         #### f1 ####
        self.conv4_3 = BasicRFB(256,256,stride = 1,scale=1.0)
        self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2)  # 16

        self.conv5_1 = BasicConv(256, 128, kernel_size=1, relu=False, bn=bn)
        self.conv5_2 = BasicConv(128, 256, kernel_size=3, padding=1, stride=1, bn=bn) #### f2 ####

        self.conv6_1 = BasicConv(256, 128, kernel_size=1, relu=False)
        self.conv6_2 = BasicConv(128, 256, kernel_size=3, padding=1, stride=2) #### f3 ####

        self.conv7_1 = BasicConv(256, 128, kernel_size=1, relu=False)
        self.conv7_2 = BasicConv(128, 256, kernel_size=3, padding=1, stride=2) #### f4 ####

        self.conv8_1 = BasicConv(256,128,kernel_size=1, relu=False)
        self.conv8_2 = BasicConv(128,256,kernel_size=3)                        #### f5 ####

        self.conv9_1 = BasicConv(256,128,kernel_size=1, relu=False)
        self.conv9_2 = BasicConv(128,256,kernel_size=3)                         #### f6 ####


    def forward(self, x):
        x = self.conv1_1(x)
        x = self.conv1_2(x)
        x = self.pool1(x)

        x = self.conv2_1(x)
        x = self.conv2_2(x)
        x = self.pool2(x)

        x = self.conv3_1(x)
        x = self.conv3_2(x)
        x = self.pool3(x)

        x = self.conv4_1(x)
        x = self.conv4_2(x)
        x = self.conv4_3(x)
        f1 = x # stride = 8
        x = self.pool4(x)

        x = self.conv5_1(x)
        x = self.conv5_2(x)
        f2 = x # stride = 16

        x = self.conv6_1(x)
        x = self.conv6_2(x)
        f3 = x # stride = 32

        x = self.conv7_1(x)
        x = self.conv7_2(x)
        f4 = x # stride = 64

        x = self.conv8_1(x)
        x = self.conv8_2(x)
        f5 = x # -2

        x = self.conv9_1(x)
        x = self.conv9_2(x)
        f6 = x # -2

        return f1, f2, f3, f4, f5, f6


if __name__ == '__main__':
    x = torch.randn(2,3,300,300)
    model = Backbone()
    features = model(x)
