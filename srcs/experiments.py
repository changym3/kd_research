
from typing import Optional
import os
import torch
import torch.nn.functional as F
from torch_geometric.nn.models import GAT
from torch_geometric.datasets import Planetoid
import torch_geometric.transforms as T
from torch_geometric.nn import GATConv

from easydict import EasyDict
import yaml

from kd.srcs.evaluator import Evaluator
from kd.trainer import Trainer


def build_model(config):
    if config.model.name == 'GAT':
        model = GAT(config.dataset.num_features, config.model.num_hiddens, config.model.num_layers, config.dataset.num_classes, 
                    jk=config.model.jk, heads=config.model.heads, dropout=config.model.dropout)
    return model

def build_dataset(config):
    if config.dataset.name == 'Cora':
        dataset = Planetoid(name='Cora', root='/home/changym/dataset', transform=T.NormalizeFeatures())
    return dataset

def build_config(model, dataset):
    config = EasyDict()
    config_dir = os.path.dirname(__file__)
    dataset_params = EasyDict(yaml.load(open(os.path.join(config_dir, 'configs/dataset_config.yaml')), Loader=yaml.FullLoader))
    model_params = EasyDict(yaml.load(open(os.path.join(config_dir, 'configs/model_config.yaml')), Loader=yaml.FullLoader))
    trainer_params = EasyDict(yaml.load(open(os.path.join(config_dir, 'configs/trainer_config.yaml')), Loader=yaml.FullLoader))
    config['dataset'] = dataset_params[dataset]
    config['model'] = model_params[model]
    config['trainer'] = trainer_params
    return config

def run_experiment(model, dataset, gpu=None, **model_kwargs):
    assert model in ['GAT']
    assert dataset in ['Cora']

    if gpu is None:
        device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    elif gpu == -1:
        device = torch.device('cpu')
    elif isinstance(gpu, int):
        device = torch.device('cuda', gpu)

    config = build_config(model, dataset)
    print(config)

    model = build_model(config)
    print(model)
    dataset = build_dataset(config)
    trainer = Trainer(config, model, dataset, device)

    trainer.fit()



if __name__ == '__main__':
    run_experiment('GAT', 'Cora', gpu=1)

#  python -m kd.experiments