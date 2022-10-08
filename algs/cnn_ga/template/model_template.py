"""
from __future__ import print_function
import mindspore
import mindspore.nn as nn
import mindspore.ops as F
import os
from datetime import datetime
import multiprocessing


class BasicBlock(nn.Cell):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, pad_mode='pad', padding=1, has_bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, pad_mode='pad', padding=1, has_bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.SequentiaCell()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion * planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion * planes)
            )

    def forward(self, x):
        out = F.ReLU(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.ReLU(out)
        return out


class EvoCNNModel(nn.Module):
    def __init__(self):
        super(EvoCNNModel, self).__init__()
        # generate_init

    def forward(self, x):
        # generate_forward

        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out

"""