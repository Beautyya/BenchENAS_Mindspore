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
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, pad_mode='pad', padding=1, has_bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.SequentialCell()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.SequentialCell(
                nn.Conv2d(in_planes, self.expansion * planes, kernel_size=1, stride=stride, has_bias=False),
                nn.BatchNorm2d(self.expansion * planes)
            )

    def construct(self, x):
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = self.relu(out)
        return out


class EvoCNNModel(nn.Cell):
    def __init__(self):
        super(EvoCNNModel, self).__init__()
        self.max_pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.avg_pool = nn.MaxPool2d(kernel_size=2, stride=2)
        # generate_init

    def construct(self, x):
        # generate_forward

        out = out.view(out.shape[0], -1)
        out = self.linear(out)
        return out

"""
