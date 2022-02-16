import torch
import torch_geometric.transforms as T
from torch_geometric.datasets import Planetoid
from ogb.nodeproppred import PygNodePropPredDataset


def build_dataset(dataset_name):
    root = '/home/changym/dataset/pyG'

    transform_list = T.Compose([
        T.ToUndirected(),
        T.AddSelfLoops(),
        T.NormalizeFeatures()
    ])

    if dataset_name in ['Cora', 'PubMed', 'CiteSeer']:
        dataset = Planetoid(name=dataset_name, root=root, transform=transform_list)
        # dataset.data['train_idx']= dataset.data.train_mask.nonzero().flatten()
        # dataset.data['val_idx']= dataset.data.val_mask.nonzero().flatten()
        # dataset.data['test_idx']= dataset.data.test_mask.nonzero().flatten()
    elif dataset_name in ['ogbn-arxiv']:
        dataset = PygNodePropPredDataset(name=dataset_name, root=root, transform=transform_list)
        splits = dataset.get_idx_split()
        train_idx, val_idx, test_idx = splits['train'], splits['valid'], splits['test']
        num_nodes = dataset[0].x.shape[0]
        dataset.data.train_mask = torch.zeros(num_nodes,dtype=torch.bool).index_fill_(0, train_idx, True)
        dataset.data.test_mask = torch.zeros(num_nodes,dtype=torch.bool).index_fill_(0, test_idx, True)
        dataset.data.val_mask = torch.zeros(num_nodes,dtype=torch.bool).index_fill_(0, val_idx, True)
    return dataset