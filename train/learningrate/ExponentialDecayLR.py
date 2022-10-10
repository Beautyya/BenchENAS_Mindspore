# coding=utf-8
from train.learningrate.learningrate import BaseLearningRate
import mindspore.nn as nn


class ExponentialDecayLR(BaseLearningRate):
    """ExponentialDecayLR
    """

    def __init__(self, **kwargs):
        super(ExponentialDecayLR, self).__init__(**kwargs)

    def get_learning_rate(self):
        return nn.ExponentialDecayLR(learning_rate=self.lr, decay_rate=0.2, decay_steps=self.total_step)

