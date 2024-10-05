import torch
import numpy as np
import random
import os
import os.path as osp
from torch.utils.tensorboard import SummaryWriter
# from .log import Logger
from datetime import datetime
import sys

def print_loss_dict(loss_dict):
    plist = []
    for k, v in loss_dict.items():
        if 'loss' in k:
            plist.append(f'{k}:{v:.4f},')
        else:
            plist.append(f'loss_{k}:{v:.4f},')
    return ' '.join(plist)


def get_time(format='%Y-%m-%d %H-%M-%S'):
    return datetime.strftime(datetime.now(), format=format)


def init_logger(cfg, use_time=True):
    """
    initialization the logger part; tensorboard, logger, save_dir
    Args:
        cfg: configs

    Returns:
        writer, logger, save_dir
    """
    if use_time:
        cfg.desc = get_time() + '  ' + cfg.desc

    os.makedirs(osp.join(cfg.dir, cfg.desc, 'tb'), exist_ok=True)
    os.makedirs(osp.join(cfg.dir, cfg.desc, 'ckpt'), exist_ok=True)

    writer = SummaryWriter(osp.join(cfg.dir, cfg.desc, 'tb'))
    logger = Logger(osp.join(cfg.dir, cfg.desc, f'{cfg.desc}_log'))
    ckpt_dir = osp.join(osp.join(cfg.dir, cfg.desc, 'ckpt'))
    return writer, logger, ckpt_dir


class DictMeter():
    def __init__(self, key=None):
        if key is None:
            self.initted = False
        else:
            self.meter = {k: 0 for k in key}
        self.count = 0

    def init_meter(self, kv_dict):
        self.meter = {k: 0 for k in kv_dict.keys()}
        self.initted = True

    def add(self, kv_dict, num):
        if not self.initted:
            self.init_meter(kv_dict)
        for k, v in kv_dict.items():
            self.meter[k] += v.sum()
        self.count += num

    def avg(self):
        for k, v in self.meter.items():
            self.meter[k] = v / self.count
        return self.meter

    def clear(self):
        self.count = 0
        self.initted = False
        self.meter = None

class ScalarMeter():
    def __init__(self):
        self.loss = 0
        self.num = 0

    def add(self, loss, num):
        self.loss += loss * num
        self.num += num

    def avg(self):
        return self.loss / self.num

    def reset(self):
        self.loss = 0
        self.num = 0


def seed_everything(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


def z_save_model(model, opt, sch, epoch, save_path):
    save_dict = {
        'submit_pt': model.state_dict(),
        'optim': opt.state_dict(),
        'sch': sch.state_dict(),
        'epoch': epoch}
    torch.save(save_dict, save_path)


def z_load_model(model, opt, sch, save_path):
    save_dict = torch.load(save_path)
    model.load_state_dict(save_dict['submit_pt'])
    opt.load_state_dict(save_dict['optim'])
    if sch is not None:
        sch.load_state_dict(save_dict['sch'])
    return save_dict['epoch']


def load_model_part(net, ckpt):
    pretrained_state_dict = torch.load(ckpt)
    filtered_state_dict = {}
    state_dict = net.state_dict()
    for k, v in pretrained_state_dict.items():
        if k in state_dict.keys():
            if v.shape == state_dict[k].shape:
                filtered_state_dict[k] = v
                print(k)

    state_dict.update(filtered_state_dict)
    net.load_state_dict(state_dict)

def init_log_as_filename(fn, prefix='train_', postfix='.py'):
    fn = os.path.basename(fn)
    name = fn.split(prefix)[-1].split(postfix)[0]
    return name

class HiddenPrints:
    """
        hidden print function to terminal
    usage:
        with HiddenPrints():
            ...
        
    """
    def __enter__(self):
        self._original_stdout = sys.stdout
        sys.stdout = open(os.devnull, 'w')

    def __exit__(self, exc_type, exc_val, exc_tb):
        sys.stdout.close()
        sys.stdout = self._original_stdout

if __name__ == '__main__':
    print(get_time())
