import logging
import os
import random
import sys
from functools import partial
from time import localtime, strftime

import numpy as np
import toml
import torch
from addict import Dict
from torch.backends import cudnn
from torchmetrics.image import (PeakSignalNoiseRatio,
                                StructuralSimilarityIndexMeasure)


def ensure_dir(path):
    os.makedirs(path, exist_ok=True)
    return path


def parse_config(path):
    return Dict(toml.load(path))


class Env(object):

    def __init__(self, args, config, training=True) -> None:
        super().__init__()

        self.device = self.setup_device(args, config)

        # experiment folder
        self._save_dir = os.path.join(config.env.exp_dir, config.env.exp_name)

        # folder to save predicted images
        self._visual_dir = os.path.join(self._save_dir, 'visual_results')

        # setup logger
        self.setup_logger(self.save_dir, save_local=training, console_high_pri_msg_only=True)

        # set seed
        if config.env.seed:
            logging.info(f'using random seed = {config.env.seed}')
            self.seed_everything(config.env.seed)

        # cudnn setup
        if config.env.deterministic:
            logging.info(f'set cudnn deterministic = True')
            cudnn.deterministic = True
            cudnn.benchmark = False

    @property
    def save_dir(self):
        return ensure_dir(self._save_dir)

    def visual_dir(self, iter):
        return ensure_dir(os.path.join(self._visual_dir, str(iter)))

    def setup_device(self, args, config):
        device = torch.device('cpu') if args.cpu or not torch.cuda.is_available() else torch.device('cuda')
        return device

    def setup_logger(self, exp_dir, save_local, console_high_pri_msg_only):
        hdls = []
        if save_local:
            log_fpath = os.path.join(exp_dir, f'{strftime("%m%d_%H%M%S", localtime())}.log')
            hdls.append(logging.FileHandler(log_fpath))

        console_hdl = logging.StreamHandler(sys.stdout)
        if console_high_pri_msg_only:
            console_hdl.setLevel(logging.INFO)
        hdls.append(console_hdl)

        logging.basicConfig(
            handlers=hdls,
            level=logging.DEBUG,
            format='%(asctime)s | %(message)s',
            datefmt='%m-%d %H:%M:%S',
            force=True
        )

    def seed_everything(self, seed):
        """ Set random seed """
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        os.environ['PYTHONHASHSEED'] = str(seed)


def init_env(args, config, training=True):
    return Env(args, config, training)


def save_state_dict(model, optimizer, curr_iter, save_path):
    torch.save({
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'start_iter': curr_iter + 1   # next iter
    }, save_path)


def init_metrics(cfg):
    METRICS = {
        # all metrics are computed in torch.uint8 data type
        'psnr': partial(PeakSignalNoiseRatio, data_range=255.0),
        'ssim': partial(StructuralSimilarityIndexMeasure, data_range=255.0)
    }
    metric_cls = METRICS.get(cfg.name.lower(), None)
    if not metric_cls:
        raise NotImplementedError

    metric = metric_cls()
    return (cfg.name, metric)