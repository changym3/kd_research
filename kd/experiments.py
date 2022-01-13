
from typing import Optional
import os
import argparse
import torch
import torch.nn.functional as F
from torch_geometric.nn.models import GAT, MLP
from torch_geometric.datasets import Planetoid
import torch_geometric.transforms as T
from torch_geometric.nn import GATConv

from easydict import EasyDict
import yaml


from kd.trainer import MLPTrainer, GNNTrainer


def build_model(config):
    num_features = config.dataset.num_features
    num_hiddens = config.model.num_hiddens
    num_layers = config.model.num_layers
    num_classes = config.dataset.num_classes
    dropout = config.model.dropout
    

    if config.model.name == 'GAT':
        model = GAT(num_features, num_hiddens, num_layers, num_classes, 
                    jk=config.model.jk, heads=config.model.heads, dropout=dropout)
    elif config.model.name == 'MLP':
        channel_list = [num_features, *([num_hiddens] * (num_layers - 1)), num_classes]
        model = MLP(channel_list, dropout, batch_norm=config.model.batch_norm)

    return model

def build_dataset(config):
    if config.dataset.name == 'Cora':
        dataset = Planetoid(name='Cora', root='/home/changym/dataset')
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

def experiment(model, dataset, gpu=None, **model_kwargs):
    assert model in ['GAT', 'MLP']
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
    
    if config.model.name == 'GAT':
        trainer = GNNTrainer(config, model, dataset, device)
    elif config.model.name == 'MLP':
        trainer = MLPTrainer(config, model, dataset, device)

    trainer.fit()



if __name__ == '__main__':
    # run_experiment('GAT', 'Cora', gpu=1)
    # run_experiment('MLP', 'Cora', gpu=1)


    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='MLP')
    parser.add_argument('--dataset', type=str, default='Cora')
    parser.add_argument('--gpu', type=int, default=1)
    
    args = parser.parse_args()
    experiment(args.model, args.dataset, gpu=args.gpu)

#  python -m kd.experiments