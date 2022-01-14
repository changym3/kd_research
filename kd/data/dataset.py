import torch_geometric.transforms as T
from torch_geometric.datasets import Planetoid


def build_dataset(dataset_name):
    if dataset_name == 'Cora':
        dataset = Planetoid(name='Cora', root='/home/changym/dataset', transform=T.NormalizeFeatures())
    return dataset