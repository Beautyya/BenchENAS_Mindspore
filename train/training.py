# coding=utf-8
from __future__ import print_function
import sys
import os

import mindspore
import traceback
import mindspore.nn as nn
import importlib, argparse
from datetime import datetime
import multiprocessing
from algs.performance_pred.utils import TrainConfig

from mindspore import context, Tensor, ops

from comm.registry import Registry
from train.utils import OptimizerConfig, LRConfig

from compute.redis import RedisLog
from compute.pid_manager import PIDManager


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
        best_acc = 0.0
        self.net = net

        TrainConfig.ConfigTrainModel(self)
        # initialize optimizer
        o = OptimizerConfig()
        opt_cls = Registry.OptimizerRegistry.query(o.read_ini_file('_name'))
        opt_params = {k: (v) for k, v in o.read_ini_file_all().items() if not k.startswith('_')}
        l = LRConfig()
        lr_cls = Registry.LRRegistry.query(l.read_ini_file('lr_strategy'))
        lr_params = {k: (v) for k, v in l.read_ini_file_all().items() if not k.startswith('_')}
        lr_params['lr'] = float(lr_params['lr'])
        opt_params['lr'] = float(lr_params['lr'])
        self.opt_params = opt_params
        self.opt_cls = opt_cls
        self.opt_params['total_epoch'] = self.nepochs
        self.lr_params = lr_params
        self.lr_cls = lr_cls
        # after the initialization

        self.best_acc = best_acc

        self.file_id = file_id
        self.logger = logger

    def log_record(self, _str):
        dt = datetime.now()
        dt.strftime('%Y-%m-%d %H:%M:%S')
        self.logger.info('[%s]-%s' % (dt, _str))

    def get_optimizer(self, epoch):
        # get optimizer
        self.opt_params['current_epoch'] = epoch
        opt_cls_ins = self.opt_cls(**self.opt_params)
        optimizer = opt_cls_ins.get_optimizer(filter(lambda p: p.requires_grad, self.net.parameters()))
        return optimizer

    def get_learning_rate(self, epoch):
        step_per_epoch = self.trainloader.get_dataset_size()
        min_lr = 0.
        max_lr = self.lr_params['lr']
        total_step = step_per_epoch * epoch
        learning_rate = Tensor(
            nn.cosine_decay_lr(max_lr=max_lr, min_lr=min_lr, total_step=total_step, step_per_epoch=step_per_epoch,
                               decay_epoch=epoch))

        return learning_rate

    def process(self):
        total_epoch = self.nepochs
        scheduler = self.get_learning_rate(total_epoch)
        self.optimizer = nn.SGD(
            params=self.net.trainable_params(),
            learning_rate=scheduler,
            momentum=0.9,
            weight_decay=0.0005,
            nesterov=True
        )
        for p in range(total_epoch):
            self.train(p)
            self.main(p)
        return self.best_acc

    def train(self, epoch):
        def forward_fn(img, labels):
            logits = self.net(img)
            losses = self.criterion(logits, labels)
            return losses, logits

        # Get gradient function
        grad_fn = ops.value_and_grad(forward_fn, None, self.optimizer.parameters, has_aux=True)

        # Define function of one-step training
        def train_step(img, labels):
            (losses, _), grads = grad_fn(img, labels)
            losses = ops.depend(losses, self.optimizer(grads))
            return losses

        self.net.set_train()
        loss = 0.
        for batch, (data, label) in enumerate(self.trainloader.create_tuple_iterator()):
            loss = train_step(data, label)

        self.log_record('Train-Epoch:%3d,  Loss: %.3f' % (epoch + 1, loss))

    def main(self, epoch):
        num_batches = self.validate_loader.get_dataset_size()
        self.net.set_train(False)
        total, test_loss, correct = 0, 0, 0
        for data, label in self.validate_loader.create_tuple_iterator():
            pred = self.net(data)
            total += len(data)
            test_loss += self.criterion(pred, label).asnumpy()
            correct += (pred.argmax(1) == label).asnumpy().sum()
        test_loss /= num_batches
        correct /= total
        if correct > self.best_acc:
            self.best_acc = correct
        self.log_record(f"Test-Epoch:{epoch} Accuracy: {correct:>0.3f}, Avg loss: {test_loss:>8f}")


class RunModel(object):

    def log_record(self, _str):
        dt = datetime.now()
        dt.strftime('%Y-%m-%d %H:%M:%S')
        self.logger.info('[%s]-%s' % (dt, _str))

    def do_work(self, gpu_id, file_id, uuid):
        os.environ['CUDA_VISIBLE_DEVICES'] = gpu_id
        context.set_context(mode=context.PYNATIVE_MODE, device_target='Ascend')
        context.set_context(device_id=0)
        logger = RedisLog(os.path.basename(file_id) + '.txt')
        best_acc = 0.0
        try:
            m = TrainModel(file_id, logger)
            m.log_record(
                'Used GPU#%s, worker name:%s[%d]' % (gpu_id, multiprocessing.current_process().name, os.getpid()))
            best_acc = m.process()
        except BaseException as e:
            msg = traceback.format_exc()
            print('Exception occurs, file:%s, pid:%d...%s' % (file_id, os.getpid(), str(e)))
            print('%s' % (msg))
            dt = datetime.now()
            dt.strftime('%Y-%m-%d %H:%M:%S')
            _str = 'Exception occurs:%s' % (msg)
            logger.info('[%s]-%s' % (dt, _str))
        finally:
            dt = datetime.now()
            dt.strftime('%Y-%m-%d %H:%M:%S')
            _str = 'Finished-Acc:%.3f' % best_acc
            logger.info('[%s]-%s' % (dt, _str))

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
