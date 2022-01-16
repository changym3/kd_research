import json
import os
from copy import deepcopy
import torch

from kd.trainer import MLPTrainer, BasicGNNTrainer, KDModelTrainer
from kd.data import build_dataset

class Experiment:
    def __init__(self, model_cfg, dataset_cfg, n_runs=1):
        self.config = self.prepare_cfg(model_cfg, dataset_cfg)
        self.device = self.build_device(self.config.trainer.gpu)
        self.dataset = build_dataset(self.config.meta.dataset_name)
        # self.trainer = self.build_trainer(self.config.meta.model_name)
        self.n_runs = n_runs

    def prepare_cfg(self, model_cfg, dataset_cfg):
        cfg = deepcopy(model_cfg)
        cfg.meta.dataset_name = dataset_cfg.name
        cfg.dataset = dataset_cfg
        return cfg

    def build_device(self, gpu):
        if gpu is None:
            device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        elif gpu == -1:
            device = torch.device('cpu')
        elif isinstance(gpu, int):
            device = torch.device('cuda', gpu)
        return device
    
    def build_trainer(self, model_name):
        if model_name == 'MLP':
            trainer = MLPTrainer(self.config, self.dataset, self.device)
        elif model_name in ['GAT', 'GCN']:
            trainer = BasicGNNTrainer(self.config, self.dataset, self.device)
        elif model_name == 'KDModel':
            trainer = KDModelTrainer(self.config, self.dataset, self.device)
        return trainer

    def run(self):
        res = []
        # result : highest_train, best_epoch, train, valid, test
        for i in range(self.n_runs):
            trainer = self.run_single()
            res.append(trainer.logger.report(verbose=False))
        
        if len(res) == 1:
            trainer.logger.report()
        else:
            r = torch.as_tensor(res)
            print(f'Highest Train: {r[:, 0].mean():.4f} ± {r[:, 0].std():.4f}')
            print(f'Best Epoch : {r[:, 1].mean():.4f} ± {r[:, 1].std():.4f}')
            print(f'Best Epoch - Train: {r[:, 2].mean():.4f} ± {r[:, 2].std():.4f}')
            print(f'Best Epoch - Valid: {r[:, 3].mean():.4f} ± {r[:, 3].std():.4f}')
            print(f'Best Epoch - Test: {r[:, 4].mean():.4f} ± {r[:, 4].std():.4f}')

        
    def run_single(self):
        trainer = self.build_trainer(self.config.meta.model_name)
        trainer.fit()
        trainer.logger.report()
        if hasattr(trainer, 'checkpoint') and trainer.checkpoint:
            print(f'The model is at epoch {trainer.checkpoint.best_epoch}, the saving directory is {os.path.realpath(trainer.checkpoint.ckpt_dir)}')
        print(json.dumps(self.config, indent=4))
        return trainer

