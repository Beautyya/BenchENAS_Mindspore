"""
from __future__ import print_function
import argparse
import mindspore
import mindspore.nn as nn
import os
import datetime 

class ConvBlock(nn.Cell):
    def __init__(self,ch_in,ch_out,kernel_size=3, stride=1):
        super(ConvBlock,self).__init__()
        self.conv_1 = nn.Conv2d(ch_in, ch_out, kernel_size, stride=stride, pad_mode='pad', padding=1)
        self.bn_1 = nn.BatchNorm2d(ch_out)
        self.relu = nn.ReLU()

    def  construct(self,x):
        out = self.relu(self.bn_1(self.conv_1(x)))
        return out



class EvoCNNModel(nn.Cell):
    def __init__(self):
        super(EvoCNNModel, self).__init__()
        self.pool = nn.MaxPool2d(3, stride=2, pad_mode='same')
        self.relu = nn.ReLU()
        self.flatten = nn.Flatten()
        self.log_softmax = nn.LogSoftmax(axis=1)
        # generate_init


    def construct(self, input):
        # generate_forward

        out = self.flatten(input)
        out = self.linear(out)
        output = self.log_softmax(out)
        return output
"""
