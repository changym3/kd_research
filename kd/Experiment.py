import json
import os
import torch


from kd.configs.config import build_config
from kd.trainer import MLPTrainer, BasicGNNTrainer, KDModelTrainer
from kd.data import build_dataset

class Experiment:
    def __init__(self, config=None, cfg_path=None):
        assert config is not None or cfg_path is not None
        if config is None:
            self.config = build_config(cfg_path)
        else:
            self.config = config

        self.dataset_name = self.config.meta.dataset_name
        self.model_name = self.config.meta.model_name
        self.device = self.build_device(self.config.trainer.gpu)
        self.dataset = build_dataset(self.config.meta.dataset_name)
        self.trainer = self.build_trainer(self.config.meta.model_name)

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
        # assert model in ['GAT', 'MLP', 'GCN']
        # assert dataset in ['Cora']

        self.trainer.fit()
        self.trainer.logger.report()
        if hasattr(self.trainer, 'checkpoint') and self.trainer.checkpoint:
            print(f'The model is at epoch {self.trainer.checkpoint.best_epoch}, the saving directory is {os.path.realpath(self.trainer.checkpoint.ckpt_dir)}')



        print(json.dumps(self.config, indent=4))

        

        

