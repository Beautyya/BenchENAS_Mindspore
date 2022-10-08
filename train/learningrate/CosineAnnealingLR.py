# coding=utf-8
from train.learningrate.learningrate import BaseLearningRate
import mindspore.nn as nn


class CosineAnnealingLR(BaseLearningRate):
    """CosineAnnealingLR
    """

    def __init__(self, **kwargs):
        super(CosineAnnealingLR, self).__init__(**kwargs)

    def get_learning_rate(self):
        return nn.cosine_decay_lr(self.optimizer, float(self.current_epoch))

