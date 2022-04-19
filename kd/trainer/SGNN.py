from collections import defaultdict
import os.path as osp
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn.models import MLP, GAT
import torch_geometric.transforms as T
from torch_geometric.utils import negative_sampling

from kd.utils.evaluator import Evaluator
from kd.utils.logger import Logger
from kd.utils.checkpoint import Checkpoint
from kd.utils.augmentation import AugmentedFeatures
from kd.utils.combine import ChannelCombine
import kd.knowledge as K


class SGNN(nn.Module):
    def __init__(self, cfg, data) -> None:
        super().__init__()
        self.cfg = cfg
        self.data = data
        self.af = AugmentedFeatures(cfg.model.aug_hop)
        self.feat_list = self.af.augment_features(self.data)

        num_channels = cfg.model.aug_hop + 1
        num_features = cfg.dataset.num_features
        num_classes = cfg.dataset.num_classes
        num_hiddens = cfg.model.num_hiddens
        num_layers = cfg.model.num_layers
        batch_norm = cfg.model.batch_norm
        if cfg.model.get('dropout', None) is None:
            inception_dropout = cfg.model.inception_dropout
            projector_dropout = cfg.model.projector_dropout
        else:
            inception_dropout = cfg.model.dropout
            projector_dropout = cfg.model.dropout

        self.inception_fcs = nn.ModuleList([
            MLP([num_features, num_hiddens, num_hiddens], dropout=inception_dropout) for _ in range(num_channels)
        ])
        self.channel_combine = ChannelCombine(
            num_hiddens, cfg.model.feat_combine, num_channels, cfg.model.get('attn_drop', 0))
        self.projection = MLP([num_hiddens] * num_layers +
                              [num_classes], dropout=projector_dropout, batch_norm=batch_norm)

    def forward_state(self):
        ks = defaultdict(list)
        hiddens = []
        for x, fc in zip(self.feat_list, self.inception_fcs):
            x = fc(x)
            hiddens.append(x)
        ks['inceptions'] = hiddens
        x = self.channel_combine(torch.stack(hiddens, dim=0))
        ks['combine'] = x
        x = F.relu(x)
        ks['projection'] = self.get_mlp_intermediate_state(self.projection, x)
        return ks

    def get_mlp_intermediate_state(self, model, x):
        states = []
        x = model.lins[0](x)
        states.append(x)
        for lin, norm in zip(model.lins[1:], model.norms):
            if model.relu_first:
                x = x.relu_()
            x = norm(x)
            if not model.relu_first:
                x = x.relu_()
            x = F.dropout(x, p=model.dropout, training=model.training)
            x = lin.forward(x)
            states.append(x)
        return states


class SGNNTrainer:
    def __init__(self, cfg, dataset, device):
        self.cfg = cfg
        self.dataset = dataset
        self.data = dataset[0].to(device)
        self.device = device
        self.model = SGNN(cfg, data=self.data).to(device)

        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=cfg.trainer.lr, weight_decay=cfg.trainer.weight_decay)
        self.evaluator = Evaluator()
        self.logger = Logger()
        if self.cfg.trainer.get('ckpt_dir', None) is None:
            self.checkpoint = None
        else:
            self.checkpoint = Checkpoint(cfg, cfg.trainer.ckpt_dir)

    def fit(self):
        for epoch in range(self.cfg.trainer.epochs):
            loss = self.train_epoch(self.model, self.data, self.optimizer)
            ks, train_acc, val_acc, test_acc = self.eval_epoch(
                self.evaluator, self.data, self.model)
            self.logger.add_result(
                epoch, loss, train_acc, val_acc, test_acc, verbose=self.cfg.trainer.verbose)
            if self.checkpoint:
                self.checkpoint.report(epoch, self.model, val_acc)
        if self.checkpoint:
            self.save_knowledge(self.model, self.data)

    def save_knowledge(self, model, data):
        ckpt_dir = self.cfg.trainer.ckpt_dir
        model.load_state_dict(torch.load(
            osp.join(ckpt_dir, 'model.pt'))['model_state_dict'])
        ks, train_acc, val_acc, test_acc = self.eval_epoch(
            self.evaluator, data, model)

        kno_path = osp.join(ckpt_dir, 'knowledge.pt')
        torch.save({'knowledge': ks}, kno_path)
        print(f'Save predictions into file {kno_path}.')
        print(
            f'The Prediction, Train: {train_acc:.4f}, Val: {val_acc:.4f}, Test: {test_acc:.4f}')

    def train_epoch(self, model, data, optimizer):
        model.train()
        optimizer.zero_grad()
        ks = model.forward_state()
        loss = F.cross_entropy(ks['projection'][-1][data.train_mask], data.y[data.train_mask])
        loss.backward()
        optimizer.step()
        return float(loss)

    @torch.no_grad()
    def eval_epoch(self, evaluator, data, model):
        model.eval()
        ks = model.forward_state()
        out = ks['projection'][-1]
        train_acc = evaluator.eval(
            out[data.train_mask], data.y[data.train_mask])['acc']
        val_acc = evaluator.eval(
            out[data.val_mask], data.y[data.val_mask])['acc']
        test_acc = evaluator.eval(
            out[data.test_mask], data.y[data.test_mask])['acc']
        return ks, train_acc, val_acc, test_acc
