import os
import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from torch_geometric.nn.models import GAT, GCN


from kd.utils.checkpoint import Checkpoint
from kd.utils.evaluator import Evaluator
from kd.utils.logger import Logger

# class GCN(torch.nn.Module):
#     def __init__(self, in_channels, hidden_channels, out_channels, num_layers,
#                  dropout):
#         super(GCN, self).__init__()

#         self.convs = torch.nn.ModuleList()
#         self.convs.append(GCNConv(in_channels, hidden_channels, cached=True))
#         self.bns = torch.nn.ModuleList()
#         self.bns.append(torch.nn.BatchNorm1d(hidden_channels))
#         for _ in range(num_layers - 2):
#             self.convs.append(
#                 GCNConv(hidden_channels, hidden_channels, cached=True))
#             self.bns.append(torch.nn.BatchNorm1d(hidden_channels))
#         self.convs.append(GCNConv(hidden_channels, out_channels, cached=True))

#         self.dropout = dropout

#     def reset_parameters(self):
#         for conv in self.convs:
#             conv.reset_parameters()
#         for bn in self.bns:
#             bn.reset_parameters()

#     def forward(self, x, adj_t):
#         for i, conv in enumerate(self.convs[:-1]):
#             x = conv(x, adj_t)
#             x = self.bns[i](x)
#             x = F.relu(x)
#             x = F.dropout(x, p=self.dropout, training=self.training)
#         x = self.convs[-1](x, adj_t)
#         return x.log_softmax(dim=-1)


class BasicGNNTrainer:
    def __init__(self, cfg, dataset, device):
        self.cfg = cfg
        self.dataset = dataset
        self.data = dataset[0].to(device)
        self.device = device
        self.model = BasicGNNTrainer.build_model(cfg).to(device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=cfg.trainer.lr, weight_decay=cfg.trainer.weight_decay)
        self.criterion = torch.nn.CrossEntropyLoss()
        self.evaluator = Evaluator()
        self.logger = Logger()

        if self.cfg.trainer.get('ckpt_dir', None) is None:
            self.checkpoint = None
        else:
            self.checkpoint = Checkpoint(cfg, cfg.trainer.ckpt_dir)
    
    def build_model(cfg):
        num_features = cfg.dataset.num_features
        num_hiddens = cfg.model.num_hiddens
        num_layers = cfg.model.num_layers
        num_classes = cfg.dataset.num_classes
        dropout = cfg.model.dropout
        jk = cfg.model.jk

        if cfg.meta.model_name == 'GAT':
            model = GAT(num_features, num_hiddens, num_layers, num_classes, 
                jk=jk, heads=cfg.model.heads, dropout=dropout)
        elif cfg.meta.model_name == 'GCN':
            model = GCN(num_features, num_hiddens, num_layers, num_classes,
                        dropout=dropout, jk=jk)
            # model = GCN(num_features, num_hiddens, num_classes, num_layers, dropout)
        return model

    def fit(self):
        for epoch in range(self.cfg.trainer.epochs):
            loss = self.train_epoch(self.model, self.data, self.optimizer, self.criterion)
            train_acc, val_acc, test_acc = self.eval_epoch(self.evaluator, self.data, model=self.model)
            self.logger.add_result(epoch, loss, train_acc, val_acc, test_acc, verbose=self.cfg.trainer.verbose)
            if self.checkpoint:
                self.checkpoint.report(epoch, self.model, val_acc)

            
    def train_epoch(self, model, data, optimizer, criterion):
        model.train()
        optimizer.zero_grad()
        out = model(data.x, data.edge_index)
        loss = criterion(out[data.train_mask], data.y[data.train_mask].view(-1))
        loss.backward()
        optimizer.step()
        return float(loss)

    #torch.no_grad()
    def eval_epoch(self, evaluator, data, out=None, model=None):
        assert out is not None or model is not None
        if out is None:
            model.eval()
            out = model(data.x, data.edge_index)
        train_acc = evaluator.eval(out[data.train_mask], data.y[data.train_mask])['acc']
        val_acc = evaluator.eval(out[data.val_mask], data.y[data.val_mask])['acc']
        test_acc = evaluator.eval(out[data.test_mask], data.y[data.test_mask])['acc']
        return train_acc, val_acc, test_acc