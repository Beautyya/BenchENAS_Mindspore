"""
from __future__ import print_function
import mindspore
from torch.autograd import Variable
import minspore.nn as nn
import mindspore.ops as F
import os,argparse
import numpy as np

class EvoCNNModel(nn.Module):
    def __init__(self):
        super(EvoCNNModel, self).__init__()
        #ANCHOR-generated_init


    def forward(self, x):
        #ANCHOR-generate_forward

        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out 

"""