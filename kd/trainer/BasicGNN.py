from collections import defaultdict
import os.path as osp
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, GATConv, SGConv
from torch_geometric.nn.models import GCN, GAT


from kd.utils.checkpoint import Checkpoint
from kd.utils.evaluator import Evaluator
from kd.utils.logger import Logger

    
def get_gnn_intermediate_state(model, x, edge_index, *args, **kwargs):
    model.eval()
    states = []
    xs = []
    for i in range(model.num_layers):
        x = model.convs[i](x, edge_index, *args, **kwargs)
        states.append(x)
        if (i == model.num_layers - 1 and model.has_out_channels
                and model.jk_mode == 'last'):
            break
        if model.norms is not None:
            x = model.norms[i](x)
        if model.act is not None:
            x = model.act(x)
        x = F.dropout(x, p=model.dropout, training=model.training)
        if hasattr(model, 'jk'):
            xs.append(x)
    x = model.jk(xs) if hasattr(model, 'jk') else x
    x = model.lin(x) if hasattr(model, 'lin') else x
    return states

@torch.no_grad()
def get_state(model, x, edge_index):
    model.eval()
    ks = defaultdict(list)
    states = get_gnn_intermediate_state(model.gnn_model, x, edge_index)
    logits = model(x, edge_index)
    states.append(logits)    
    states = [s.to('cpu') for s in states]
    ks['feats'] = states
    return ks

    
class GNNModel(nn.Module):
    def __init__(self, cfg) -> None:
        super().__init__()
        self.cfg = cfg
        self.gnn_model = self.build_gnn_model(cfg)
        self.projector = nn.Sequential(
            # nn.BatchNorm1d(cfg.model.num_hiddens),
            # nn.ReLU(),
            nn.Dropout(cfg.model.dropout),
            nn.Linear(cfg.model.num_hiddens, cfg.dataset.num_classes, bias=False)
        )

    def build_gnn_model(self, cfg):
        num_features = cfg.dataset.num_features
        num_hiddens = cfg.model.num_hiddens
        num_layers = cfg.model.num_layers
        dropout = cfg.model.dropout
        jk = cfg.model.get('jk', 'last')
        heads = cfg.model.get('heads', 1)
        norm = cfg.model.get('norm', None)
        if norm == 'bn':
            norm = torch.nn.BatchNorm1d(num_hiddens)
        elif norm is None:
            pass
        else:
            raise Exception(f'Invalid argument "{norm}" for cfg.model.norm')

        if cfg.meta.model_name == 'GAT':
            gnn_model = GAT(num_features, num_hiddens, num_layers, num_hiddens,
                            jk=jk, heads=heads, dropout=dropout)
        elif cfg.meta.model_name == 'GCN':
            gnn_model = GCN(num_features, num_hiddens, num_layers, num_hiddens,
                            dropout=dropout, jk=jk, cached=True, norm=norm)
        elif cfg.meta.model_name == 'SGC':
            gnn_model = SGConv(num_features, num_hiddens, K=num_layers, cached=True)
        return gnn_model
    
    def forward(self, x, edge_index):
        out = self.gnn_model(x, edge_index)
        out = self.projector(out)
        return out

class BasicGNNTrainer:
    def __init__(self, cfg, dataset, device):
        time_start=time.time()
        self.cfg = cfg
        self.dataset = dataset
        self.data = dataset[0].to(device)
        self.device = device
        self.model = GNNModel(cfg).to(device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=cfg.trainer.lr, 
                                          weight_decay=cfg.trainer.weight_decay)
        self.criterion = torch.nn.CrossEntropyLoss()
        self.evaluator = Evaluator()
        self.logger = Logger()
        if self.cfg.trainer.get('ckpt_dir', None) is None:
            self.checkpoint = None
        else:
            self.checkpoint = Checkpoint(cfg, cfg.trainer.ckpt_dir)
        time_end=time.time()
        print('Preprocessing Time:', time_end - time_start, 'seconds')

    def fit(self):
        train_time = 0
        inference_time = 0
        for epoch in range(self.cfg.trainer.epochs):
            time_start=time.time()
            loss = self.train_epoch(
                self.model, self.data, self.optimizer, self.criterion)
            time_end=time.time()
            train_time += (time_end - time_start)
            
            time_start=time.time()
            train_acc, val_acc, test_acc = self.eval_epoch(self.evaluator, self.data, model=self.model)
            time_end=time.time()
            inference_time += (time_end - time_start)
            self.logger.add_result(
                epoch, loss, train_acc, val_acc, test_acc, verbose=self.cfg.trainer.verbose)
            if self.checkpoint:
                self.checkpoint.report(epoch, self.model, val_acc)

        print('Training Time:', train_time, 'seconds')
        print('Inference Time:', inference_time, 'seconds')
        if self.checkpoint:
            self.save_knowledge(self.model, self.data)
    
    def save_knowledge(self, model, data):
        ckpt_dir = self.cfg.trainer.ckpt_dir
        model.load_state_dict(torch.load(
            osp.join(ckpt_dir, 'model.pt'))['model_state_dict'])
        train_acc, val_acc, test_acc = self.eval_epoch(self.evaluator, data, model=model)
        ks = get_state(model, data.x, data.edge_index)
        
        kno_path = osp.join(ckpt_dir, 'knowledge.pt')
        torch.save({'knowledge': ks}, kno_path)
        print(f'Save predictions into file {kno_path}.')
        print(
            f'The Prediction, Train: {train_acc:.4f}, Val: {val_acc:.4f}, Test: {test_acc:.4f}')

    def train_epoch(self, model, data, optimizer, criterion):
        model.train()
        optimizer.zero_grad()
        out = model(data.x, data.edge_index)
        loss = criterion(out[data.train_mask],
                         data.y[data.train_mask].view(-1))
        loss.backward()
        optimizer.step()
        return float(loss)

    @torch.no_grad()
    def eval_epoch(self, evaluator, data, model):
        model.eval()
        out = model(data.x, data.edge_index)
        train_acc = evaluator.eval(
            out[data.train_mask], data.y[data.train_mask])['acc']
        val_acc = evaluator.eval(
            out[data.val_mask], data.y[data.val_mask])['acc']
        test_acc = evaluator.eval(
            out[data.test_mask], data.y[data.test_mask])['acc']
        return train_acc, val_acc, test_acc
