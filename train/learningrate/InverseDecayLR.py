# coding=utf-8
from train.learningrate.learningrate import BaseLearningRate
import mindspore.nn as nn


class InverseDecayLR(BaseLearningRate):
    """InverseDecayLR
    """

    def __init__(self, **kwargs):
        super(InverseDecayLR, self).__init__(**kwargs)

    def get_learning_rate(self):
        return nn.InverseDecayLR(learning_rate=self.lr, decay_rate=0.2, decay_steps=self.total_step)

