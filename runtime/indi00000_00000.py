"""
2022-03-25  21:47:39
"""
from __future__ import print_function
import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import torch.optim as optim
import os
from datetime import datetime
import multiprocessing


import torch
import torch.nn as nn
from torch.nn import init
from torch.autograd import Variable
from collections import OrderedDict
import math
import copy
import torch.nn.functional as F
import sys


class ConvBlock(nn.Module):
    def __init__(self, in_size, out_size, kernel, stride):
        super(ConvBlock, self).__init__()
        pad_size = kernel // 2
        self.conv1 = nn.Sequential(nn.Conv2d(in_size, out_size, kernel, stride=stride, padding=pad_size, bias=False),
                                       nn.BatchNorm2d(out_size),
                                       nn.ReLU(inplace=True),)

    def forward(self, inputs):
        outputs = self.conv1(inputs)
        return outputs

class DeConvBlock(nn.Module):
    def __init__(self, in_size, out_size, kernel):
        super(DeConvBlock, self).__init__()
        pad_size = kernel // 2
        self.conv1 = nn.Sequential(nn.ConvTranspose2d(in_size, out_size, kernel, stride=2, padding=pad_size, output_padding=1, bias=False),
                                       nn.BatchNorm2d(out_size),
                                       nn.ReLU(inplace=True),)

    def forward(self, inputs):
        outputs = self.conv1(inputs)
        return outputs

class ConvBlock_last(nn.Module):
    def __init__(self, in_size, out_size, kernel):
        super(ConvBlock_last, self).__init__()
        pad_size = kernel // 2
        self.conv1 = nn.Sequential(nn.Conv2d(in_size, out_size, kernel, padding=pad_size, bias=False))
                                       # nn.BatchNorm2d(out_size),
                                       # nn.Tanh())

    def forward(self, inputs):
        outputs = self.conv1(inputs)
        return outputs

class DeConvBlock_last(nn.Module):
    def __init__(self, in_size, out_size, kernel):
        super(DeConvBlock_last, self).__init__()
        pad_size = kernel // 2
        self.conv1 = nn.Sequential(nn.ConvTranspose2d(in_size, out_size, kernel, padding=pad_size, bias=False))
                                       # nn.BatchNorm2d(out_size),
                                       # nn.Tanh())

    def forward(self, inputs):
        outputs = self.conv1(inputs)
        return outputs

class ConvBlock_s(nn.Module):
    def __init__(self, in_size, out_size, kernel, stride):
        super(ConvBlock_s, self).__init__()
        pad_size = kernel // 2
        self.conv1 = nn.Sequential(nn.Conv2d(in_size, out_size, kernel, stride=stride, padding=pad_size, bias=False),
                                       nn.BatchNorm2d(out_size),
                                       nn.ReLU(inplace=True),)

    def forward(self, inputs):
        outputs = self.conv1(inputs)
        return outputs

class ConvBlock_sum(nn.Module):
    def __init__(self, in_size, out_size, kernel):
        super(ConvBlock_sum, self).__init__()
        pad_size = kernel // 2
        self.conv1 = nn.Sequential(nn.Conv2d(in_size, out_size, kernel, padding=pad_size, bias=False),
                                       nn.BatchNorm2d(out_size),
                                       nn.ReLU(inplace=True),)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, inputs1, inputs2):
        outputs = self.conv1(inputs1)
        in_data = [outputs, inputs2]
        # check of the channel size
        small_ch_id, large_ch_id = (0, 1) if in_data[0].size(1) < in_data[1].size(1) else (1, 0)
        offset = int(in_data[large_ch_id].size()[1] - in_data[small_ch_id].size()[1])
        if offset != 0:
            tmp = in_data[large_ch_id].data[:, :offset, :, :]
            tmp = Variable(tmp).clone()
            in_data[small_ch_id] = torch.cat([in_data[small_ch_id], tmp * 0], 1)
        out = torch.add(in_data[0], in_data[1])
        return self.relu(out)

class ResBlock(nn.Module):
    def __init__(self, in_size, out_size, kernel, stride):
        super(ResBlock, self).__init__()
        pad_size = kernel // 2
        self.conv1 = nn.Sequential(nn.Conv2d(in_size, out_size, kernel, stride=stride, padding=pad_size, bias=False),
                                       nn.BatchNorm2d(out_size),
                                       nn.ReLU(inplace=True),
                                       nn.Conv2d(out_size, out_size, kernel, stride=stride, padding=pad_size, bias=False),
                                       nn.BatchNorm2d(out_size))
        self.relu = nn.ReLU(inplace=True)

    def forward(self, inputs1, inputs2):
        x = self.conv1(inputs1)
        in_data = [x, inputs2]
        # # check of the image size
        # if (in_data[0].size(2) - in_data[1].size(2)) != 0:
        #     small_in_id, large_in_id = (0, 1) if in_data[0].size(2) < in_data[1].size(2) else (1, 0)
        #     pool_num = math.floor(in_data[large_in_id].size(2) / in_data[small_in_id].size(2))
        #     for _ in range(pool_num-1):
        #         in_data[large_in_id] = F.max_pool2d(in_data[large_in_id], 2, 2, 0)

        # check of the channel size
        small_ch_id, large_ch_id = (0, 1) if in_data[0].size(1) < in_data[1].size(1) else (1, 0)
        offset = int(in_data[large_ch_id].size()[1] - in_data[small_ch_id].size()[1])
        if offset != 0:
            tmp = in_data[large_ch_id].data[:, :offset, :, :]
            tmp = Variable(tmp).clone()
            in_data[small_ch_id] = torch.cat([in_data[small_ch_id], tmp * 0], 1)
        out = torch.add(in_data[0], in_data[1])
        return self.relu(out)

class Sum(nn.Module):
    def __init__(self):
        super(Sum, self).__init__()

    def forward(self, inputs1, inputs2):
        in_data = [inputs1, inputs2]
        # check of the image size
        if (in_data[0].size(2) - in_data[1].size(2)) != 0:
            small_in_id, large_in_id = (0, 1) if in_data[0].size(2) < in_data[1].size(2) else (1, 0)
            pool_num = math.floor(in_data[large_in_id].size(2) / in_data[small_in_id].size(2))
            for _ in range(pool_num-1):
                in_data[large_in_id] = F.max_pool2d(in_data[large_in_id], 2, 2, 0)
        # check of the channel size
        small_ch_id, large_ch_id = (0, 1) if in_data[0].size(1) < in_data[1].size(1) else (1, 0)
        offset = int(in_data[large_ch_id].size()[1] - in_data[small_ch_id].size()[1])
        if offset != 0:
            tmp = in_data[large_ch_id].data[:, :offset, :, :]
            tmp = Variable(tmp).clone()
            in_data[small_ch_id] = torch.cat([in_data[small_ch_id], tmp * 0], 1)
        out = torch.add(in_data[0], in_data[1])
        return out

class Concat(nn.Module):
    def __init__(self):
        super(Concat, self).__init__()

    def forward(self, inputs1, inputs2):
        in_data = [inputs1, inputs2]
        # check of the image size
        if (in_data[0].size(2) - in_data[1].size(2)) != 0:
            small_in_id, large_in_id = (0, 1) if in_data[0].size(2) < in_data[1].size(2) else (1, 0)
            pool_num = math.floor(in_data[large_in_id].size(2) / in_data[small_in_id].size(2))
            for _ in range(pool_num-1):
                in_data[large_in_id] = F.max_pool2d(in_data[large_in_id], 2, 2, 0)
        return torch.cat([in_data[0], in_data[1]], 1)

class DeConvBlock_sum(nn.Module):
    def __init__(self, in_size, out_size, kernel):
        super(DeConvBlock_sum, self).__init__()
        pad_size = kernel // 2
        self.conv1 = nn.Sequential(nn.ConvTranspose2d(in_size, out_size, kernel, stride=2, padding=pad_size, output_padding=1, bias=False),
                                       nn.BatchNorm2d(out_size),
                                       nn.ReLU(inplace=True),)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, inputs1, inputs2):
        outputs1 = self.conv1(inputs1)
        offset = outputs1.size()[2] - inputs2.size()[2]
        padding = 2 * [offset // 2, offset // 2]
        outputs2 = F.pad(inputs2, padding)
        out = torch.add(outputs1, outputs2)
        return self.relu(out)



class CGP2CNN(nn.Module):
    def __init__(self, cgp, in_channel, n_class, imgSize):
        super(CGP2CNN, self).__init__()
        self.cgp = cgp
        self.pool_size = 2
        self.arch = OrderedDict()
        self.encode = []
        # self.channel_num = [None for _ in range(len(self.cgp))]
        # self.size = [None for _ in range(len(self.cgp))]
        self.channel_num = [None for _ in range(500)]
        self.size = [None for _ in range(500)]
        self.channel_num[0] = in_channel
        self.size[0] = imgSize
        # encoder
        i = 0
        for name, in1, in2 in self.cgp:
            if name == 'input' in name:
                i += 1
                continue
            elif name == 'full':
                self.encode.append(nn.Linear(self.channel_num[in1]*self.size[in1]*self.size[in1], n_class))
            elif name == 'Max_Pool' or name == 'Avg_Pool':
                self.channel_num[i] = self.channel_num[in1]
                self.size[i] = int(self.size[in1] / 2)
                key = name.split('_')
                func =     key[0]
                if func == 'Max':
                    self.encode.append(nn.MaxPool2d(2,2))
                else:
                    self.encode.append(nn.AvgPool2d(2,2))
            elif name == 'Concat':
                self.channel_num[i] = self.channel_num[in1] + self.channel_num[in2]
                small_in_id, large_in_id = (in1, in2) if self.size[in1] < self.size[in2] else (in2, in1)
                self.size[i] = self.size[small_in_id]
                self.encode.append(Concat())
            elif name == 'Sum':
                small_in_id, large_in_id = (in1, in2) if self.channel_num[in1] < self.channel_num[in2] else (in2, in1)
                self.channel_num[i] = self.channel_num[large_in_id]
                small_in_id, large_in_id = (in1, in2) if self.size[in1] < self.size[in2] else (in2, in1)
                self.size[i] = self.size[small_in_id]
                self.encode.append(Sum())
            else:
                key = name.split('_')
                down =     key[0]
                func =     key[1]
                out_size = int(key[2])
                kernel   = int(key[3])
                if down == 'S':
                    if func == 'ConvBlock':
                        self.channel_num[i] = out_size
                        self.size[i] = self.size[in1]
                        self.encode.append(ConvBlock(self.channel_num[in1], out_size, kernel, stride=1))
                    else:
                        in_data = [out_size, self.channel_num[in1]]
                        small_in_id, large_in_id = (0, 1) if in_data[0] < in_data[1] else (1, 0)
                        self.channel_num[i] = in_data[large_in_id]
                        # small_in_id, large_in_id = (in1, in2) if self.size[in1] < self.size[in2] else (in2, in1)
                        # self.size[i] = self.size[small_in_id]
                        self.size[i] = self.size[in1]
                        self.encode.append(ResBlock(self.channel_num[in1], out_size, kernel, stride=1))
                else:
                    sys.exit('error')
                    # if func == 'ConvBlock':
                    #     self.channel_num[i] = out_size
                    #     self.size[i] = int(self.size[in1]/2)
                    #     self.encode.append(ConvBlock(self.channel_num[in1], out_size, kernel, stride=2))
                    # else:
                    #     in_data = [out_size, self.channel_num[in2]]
                    #     small_in_id, large_in_id = (in1, in2) if self.channel_num[in1] < self.channel_num[in2] else (in2, in1)
                    #     self.channel_num[i] = self.channel_num[large_in_id]
                    #     small_in_id, large_in_id = (in1, in2) if self.size[in1] < self.size[in2] else (in2, in1)
                    #     self.size[i] = self.size[small_in_id]
                    #     self.encode.append(ResBlock(self.channel_num[in1], out_size, kernel, stride=1))
            i += 1

        self.layer_module = nn.ModuleList(self.encode)
        self.outputs = [None for _ in range(len(self.cgp))]

    def main(self,x):
        outputs = self.outputs
        outputs[0] = x    # input image
        nodeID = 1
        for layer in self.layer_module:
            if isinstance(layer, ConvBlock):
                outputs[nodeID] = layer(outputs[self.cgp[nodeID][1]])
            elif isinstance(layer, ResBlock):
                outputs[nodeID] = layer(outputs[self.cgp[nodeID][1]], outputs[self.cgp[nodeID][1]])
            elif isinstance(layer, torch.nn.modules.linear.Linear):
                tmp = outputs[self.cgp[nodeID][1]].view(outputs[self.cgp[nodeID][1]].size(0), -1)
                outputs[nodeID] = layer(tmp)
            elif isinstance(layer, torch.nn.modules.pooling.MaxPool2d) or isinstance(layer, torch.nn.modules.pooling.AvgPool2d):
                if outputs[self.cgp[nodeID][1]].size(2) > 1:
                    outputs[nodeID] = layer(outputs[self.cgp[nodeID][1]])
                else:
                    outputs[nodeID] = outputs[self.cgp[nodeID][1]]
            elif isinstance(layer, Concat):
                outputs[nodeID] = layer(outputs[self.cgp[nodeID][1]], outputs[self.cgp[nodeID][2]])
            elif isinstance(layer, Sum):
                outputs[nodeID] = layer(outputs[self.cgp[nodeID][1]], outputs[self.cgp[nodeID][2]])
            else:
                sys.exit("Error at CGP2CNN forward")
            nodeID += 1
        return outputs[nodeID-1]

    def forward(self, x):
        return self.main(x)


class EvoCNNModel(nn.Module):
    def __init__(self):
        super(EvoCNNModel, self).__init__()
        cgp = [['input', 0, 0], ['S_ResBlock_64_3', 0, 0], ['Max_Pool', 0, 0], ['S_ResBlock_32_3', 1, 1], ['S_ConvBlock_64_5', 2, 1], ['S_ConvBlock_128_3', 3, 3], ['S_ConvBlock_128_5', 5, 0], ['S_ConvBlock_64_1', 4, 4], ['S_ConvBlock_32_3', 6, 5], ['Avg_Pool', 8, 2], ['S_ResBlock_64_5', 7, 8], ['Avg_Pool', 9, 7], ['Sum', 11, 10], ['S_ConvBlock_32_1', 12, 11], ['full', 13, 13]]

        in_channel = 3
        n_class = 10
        imgSize = 32
        self.cgp2cnn = CGP2CNN(cgp, in_channel, n_class, imgSize)


    def forward(self, x):
        out = self.cgp2cnn(x)
        return out
