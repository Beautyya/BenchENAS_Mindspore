# coding=utf-8
from train.learningrate.learningrate import BaseLearningRate
import mindspore.nn as nn


class CosineDecayLR(BaseLearningRate):
    """CosineDecayLR
    """

    def __init__(self, **kwargs):
        super(CosineDecayLR, self).__init__(**kwargs)

    def get_learning_rate(self):
        return nn.CosineDecayLR(min_lr=0.0001, max_lr=self.lr, decay_steps=self.total_step)

