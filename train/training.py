# coding=utf-8
from __future__ import print_function

import argparse
import importlib
import multiprocessing
import os
import sys
import traceback
from datetime import datetime

import mindspore.nn as nn
from mindspore import context, Model
from mindspore.train.callback import Callback

from algs.performance_pred.utils import TrainConfig
from comm.registry import Registry
from compute.pid_manager import PIDManager
from compute.redis import RedisLog
from train.utils import OptimizerConfig, LRConfig


def log_record(logger, _str):
    dt = datetime.now()
    dt.strftime('%Y-%m-%d %H:%M:%S')
    logger.info('[%s]-%s' % (dt, _str))


class SaveEvalCallback(Callback):
    def __init__(self, eval_function, eval_parameter_dict, epochs, logger):
        self.eval_func = eval_function
        self.eval_param = eval_parameter_dict
        self.epoch = epochs
        self.initial_epoch = 0
        self.best_acc = 0.
        self.logger = logger

    def epoch_end(self, run_context):
        cb_params = run_context.original_args()
        epoch_num = cb_params.cur_epoch_num
        loss = cb_params.net_outputs

        if epoch_num >= self.initial_epoch:
            res = self.eval_func(self.eval_param)
            log_record(self.logger, f'Epoch: train loss: {loss}, val acc: {res}')
            if res > self.best_acc:
                self.best_acc = res

    def end(self, run_context):
        log_record(self.logger, f'Finished Accuracy: {self.best_acc}')


class TrainModel(object):
    def __init__(self, file_id, logger):

        # module_name = 'scripts.%s'%(file_name)
        module_name = file_id
        if module_name in sys.modules.keys():
            self.log_record('Module:%s has been loaded, delete it' % (module_name))
            del sys.modules[module_name]
            _module = importlib.import_module('.', module_name)
        else:
            _module = importlib.import_module('.', module_name)

        net = _module.EvoCNNModel()

        # net = net.cuda()
        self.criterion = nn.SoftmaxCrossEntropyWithLogits(sparse=True, reduction='mean')
        self.net = net

        TrainConfig.ConfigTrainModel(self)
        # initialize optimizer
        o = OptimizerConfig()
        opt_cls = Registry.OptimizerRegistry.query(o.read_ini_file('_name'))
        opt_params = {k: (v) for k, v in o.read_ini_file_all().items() if not k.startswith('_')}
        l = LRConfig()
        lr_cls = Registry.LRRegistry.query(l.read_ini_file('lr_schedule'))
        lr_params = {k: (v) for k, v in l.read_ini_file_all().items() if not k.startswith('_')}
        lr_params['lr'] = float(lr_params['lr'])
        opt_params['lr'] = float(lr_params['lr'])
        self.opt_params = opt_params
        self.opt_cls = opt_cls
        self.opt_params['total_epoch'] = self.nepochs
        self.lr_params = lr_params
        self.lr_cls = lr_cls
        # after the initialization

        self.file_id = file_id
        self.logger = logger

    def apply_eval(self, eval_param):
        eval_model = eval_param['model']
        eval_ds = eval_param['dataset']
        metrics_name = eval_param['metric']
        res = eval_model.eval(eval_ds)
        return res[metrics_name]

    def get_optimizer(self, lr_schedule):
        # get optimizer
        self.opt_params['lr_schedule'] = lr_schedule
        opt_cls_ins = self.opt_cls(**self.opt_params)
        optimizer = opt_cls_ins.get_optimizer(self.net.trainable_params())  
        return optimizer

    def get_learning_rate(self):
        epoch_per_step = self.train_loader.get_dataset_size()
        self.lr_params['total_step'] = self.opt_params['total_epoch'] * epoch_per_step
        lr_cls_ins = self.lr_cls(**self.lr_params)
        learning_rate = lr_cls_ins.get_learning_rate()
        return learning_rate

    def process(self):
        # epoch_per_step = self.train_loader.get_dataset_size()
        # total_step = self.opt_params['total_epoch'] * epoch_per_step
        lr = self.get_learning_rate()
        optim = self.get_optimizer(lr)
        model = Model(self.net, loss_fn=self.criterion, optimizer=optim, metrics={'accuracy'})
        eval_param_dict = {'model': model, 'dataset': self.valid_loader, 'metric': 'accuracy'}

        save_info = SaveEvalCallback(self.apply_eval, eval_param_dict, self.opt_params['total_epoch'], self.logger)
        model.train(self.opt_params['total_epoch'], self.train_loader, callbacks=[save_info])


class RunModel(object):
    def do_work(self, gpu_id, file_id, uuid):
        os.environ['CUDA_VISIBLE_DEVICES'] = gpu_id
        # context.set_context(mode=context.PYNATIVE_MODE, device_target='Ascend')
        context.set_context(device_id=0)
        logger = RedisLog(os.path.basename(file_id) + '.txt')
        best_acc = 0.0
        try:
            m = TrainModel(file_id, logger)
            log_record(logger, 'Used GPU#%s, worker name:%s[%d]' % (
                           gpu_id, multiprocessing.current_process().name, os.getpid()))
            m.process()
        except BaseException as e:
            msg = traceback.format_exc()
            print(logger, 'Exception occurs, file:%s, pid:%d...%s' % (file_id, os.getpid(), str(e)))
            print('%s' % msg)
            dt = datetime.now()
            dt.strftime('%Y-%m-%d %H:%M:%S')
            _str = f'Exception occurs:{msg}'
            log_record(logger, '[%s]-%s' % (dt, _str))
        finally:
            logger.write_file('RESULTS', 'results.txt', '%s=%.5f\n' % (file_id, best_acc))
            _str = '%s;%.5f\n' % (uuid, best_acc)
            logger.write_file('CACHE', 'cache.txt', _str)


if __name__ == "__main__":
    _parser = argparse.ArgumentParser()
    _parser.add_argument("-gpu_id", "--gpu_id", help="GPU ID", type=str)
    _parser.add_argument("-file_id", "--file_id", help="file id", type=str)
    _parser.add_argument("-uuid", "--uuid", help="uuid of the individual", type=str)

    _parser.add_argument("-super_node_ip", "--super_node_ip", help="ip of the super node", type=str)
    _parser.add_argument("-super_node_pid", "--super_node_pid", help="pid on the super node", type=int)
    _parser.add_argument("-worker_node_ip", "--worker_node_ip", help="ip of this worker node", type=str)

    _args = _parser.parse_args()

    PIDManager.WorkerEnd.add_worker_pid(_args.super_node_ip, _args.super_node_pid, _args.worker_node_ip)

    RunModel().do_work(_args.gpu_id, _args.file_id, _args.uuid)

    PIDManager.WorkerEnd.remove_worker_pid(_args.super_node_ip, _args.super_node_pid, _args.worker_node_ip)
