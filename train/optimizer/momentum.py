# coding=utf-8

from train.optimizer.optimizer import BaseOptimizer
import mindspore.nn as nn


class Momentum(BaseOptimizer):
    """SGD optimizer
    """

    def __init__(self, **kwargs):
        super(Momentum, self).__init__(**kwargs)

    def get_optimizer(self, weight_params):
        return nn.Momentum(weight_params, learning_rate=self.lr_schedule, momentum=0.9, weight_decay=1e-5)
