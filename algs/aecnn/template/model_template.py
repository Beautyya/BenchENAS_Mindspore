"""
from __future__ import print_function
import mindspore
import mindspore.nn as nn

class ResNetBottleneck(nn.Cell):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(ResNetBottleneck, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, has_bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, pad_mode='pad', padding=1, has_bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, self.expansion*planes, kernel_size=1, has_bias=False)
        self.bn3 = nn.BatchNorm2d(self.expansion*planes)

        self.shortcut = nn.SequentialCell()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.SequentialCell(
                nn.Conv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, has_bias=False),
                nn.BatchNorm2d(self.expansion*planes)
            )

    def construct(self, x):
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out += self.shortcut(x)
        out = self.relu(out)
        return out

class ResNetUnit(nn.Cell):
    def __init__(self, amount, in_channel, out_channel):
        super(ResNetUnit, self).__init__()
        self.in_planes = in_channel
        self.layer = self._make_layer(ResNetBottleneck, out_channel, amount, stride=1)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.SequentialCell(*layers)
    def construct(self, x):
        out = self.layer(x)
        return out

class DenseNetBottleneck(nn.Cell):
    def __init__(self, nChannels, growthRate):
        super(DenseNetBottleneck, self).__init__()
        interChannels = 4*growthRate
        self.bn1 = nn.BatchNorm2d(nChannels)
        self.conv1 = nn.Conv2d(nChannels, interChannels, kernel_size=1,
                               has_bias=False)
        self.relu = nn.ReLU()
        self.bn2 = nn.BatchNorm2d(interChannels)
        self.conv2 = nn.Conv2d(interChannels, growthRate, kernel_size=3, pad_mode='pad',
                               padding=1, has_bias=False)

    def construct(self, x):
        out = self.conv1(self.relu(self.bn1(x)))
        out = self.conv2(self.relu(self.bn2(out)))
        out = mindspore.ops.Concat((x, out), 1)
        return out

class DenseNetUnit(nn.Cell):
    def __init__(self, k, amount, in_channel, out_channel, max_input_channel):
        super(DenseNetUnit, self).__init__()
        self.out_channel = out_channel
        if in_channel > max_input_channel:
            self.need_conv = True
            self.bn = nn.BatchNorm2d(in_channel)
            self.conv = nn.Conv2d(in_channel, max_input_channel, kernel_size=1, has_bias=False)
            in_channel = max_input_channel
        self.relu = nn.ReLU
        self.layer = self._make_dense(in_channel, k, amount)

    def _make_dense(self, nChannels, growthRate, nDenseBlocks):
        layers = []
        for _ in range(int(nDenseBlocks)):
            layers.append(DenseNetBottleneck(nChannels, growthRate))
            nChannels += growthRate
        return nn.SequentialCell(*layers)
    def construct(self, x):
        out = x
        if hasattr(self, 'need_conv'):
            out = self.conv(self.relu(self.bn(out)))
        out = self.layer(out)
        # assert(out.shape[1] == self.out_channel)
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
