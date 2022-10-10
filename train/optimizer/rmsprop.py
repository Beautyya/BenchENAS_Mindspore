# coding=utf-8

from train.optimizer.optimizer import BaseOptimizer
import mindspore.nn as nn


class RMSprop(BaseOptimizer):
    """RMSprop optimizer
    """

    def __init__(self, **kwargs):
        super(RMSprop, self).__init__(**kwargs)

    def get_optimizer(self, weight_params):
        return nn.RMSProp(weight_params, learning_rate=self.lr_schedule, weight_decay=1e-5)
