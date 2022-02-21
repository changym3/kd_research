import os.path as osp
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn.models import MLP, GAT
from torch_geometric.utils import negative_sampling

from kd.utils.evaluator import Evaluator
from kd.utils.logger import Logger
import kd.knowledge as K


class KDModelTrainer:
    def __init__(self, cfg, dataset, device):
        self.cfg = cfg
        self.dataset = dataset
        self.data = dataset[0].to(device)
        self.device = device
        self.model = self.build_model(cfg).to(device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=cfg.trainer.lr, weight_decay=cfg.trainer.weight_decay)
        self.evaluator = Evaluator()
        self.logger = Logger()

        self.kd_cfg = cfg.trainer.kd
        self.kd_module = KDModule(self.cfg, verbose=cfg.trainer.verbose, device=device)
        self.knowledge = self.setup_knowledge(osp.join(self.kd_cfg.knowledge_dir, 'knowledge.pt'), self.device)


    def build_model(self, cfg):
        num_features = cfg.dataset.num_features
        num_classes = cfg.dataset.num_classes
        
        cfgm = cfg.model
        num_hiddens = cfgm.num_hiddens
        num_layers = cfgm.num_layers
        dropout = cfgm.dropout

        if cfg.meta.student_name == 'MLP':
            batch_norm = cfgm.batch_norm
            channel_list = [num_features, *([num_hiddens] * (num_layers - 1)), num_classes]
            model = MLP(channel_list, dropout, batch_norm=batch_norm)
        elif cfg.meta.student_name == 'GAT':
            jk = cfgm.jk
            heads = cfgm.heads
            model = GAT(num_features, num_hiddens, num_layers, num_classes, 
                jk=jk, heads=heads, dropout=dropout)
        return model

    def model_forward(self, model, data):
        if self.cfg.meta.student_name == 'MLP':
            return model(data.x)
        else:
            return model(data.x, data.edge_index)

    def fit(self):
        for epoch in range(self.cfg.trainer.epochs):
            loss = self.train_epoch(self.model, self.data, self.optimizer)
            train_acc, val_acc, test_acc = self.eval_epoch(self.evaluator, self.data, model=self.model)
            self.logger.add_result(epoch, loss, train_acc, val_acc, test_acc, verbose=self.cfg.trainer.verbose)

    def kd_loss(self, outs, data):
        loss = self.kd_module.loss(outs, self.knowledge, data)
        return loss
    
    def train_epoch(self, model, data, optimizer):
        model.train()
        optimizer.zero_grad()
        outs = K.get_model_state(model, data, self.cfg.meta.student_name)
        loss = self.kd_loss(outs, data)
        loss.backward()
        optimizer.step()
        return float(loss)

    #torch.no_grad()
    def eval_epoch(self, evaluator, data, out=None, model=None):
        assert out is not None or model is not None
        if out is None:
            model.eval()
            out = self.model_forward(model, data)
        train_acc = evaluator.eval(out[data.train_mask], data.y[data.train_mask])['acc']
        val_acc = evaluator.eval(out[data.val_mask], data.y[data.val_mask])['acc']
        test_acc = evaluator.eval(out[data.test_mask], data.y[data.test_mask])['acc']
        return train_acc, val_acc, test_acc
    
    def setup_knowledge(self, kno_path, device):
        knowledge = torch.load(kno_path)['knowledge'].to(device)
        return knowledge



class KDModule:
    def __init__(self, total_cfg, verbose=None, device='cuda:0') -> None:
        self.verbose = verbose
        self.total_cfg = total_cfg
        self.cfg = total_cfg.trainer.kd

        self.method = self.cfg.method
        self.mask = self.cfg.mask
        self.T = self.cfg.temperature
        self.alpha = self.cfg.get('alpha')
        self.beta = self.cfg.get('beta')

        self.link_predictor = LinkPredictor(self.total_cfg.model.num_hiddens, self.cfg.get('neg_k')).to(device)

    def loss(self, outs, knowledge, data):
        y, train_mask, val_mask, test_mask = data.y, data.train_mask, data.val_mask, data.test_mask
        assert len(train_mask) == len(val_mask) and len(val_mask) == len(test_mask)
        if self.mask == 'all':
            mask = torch.ones_like(train_mask, dtype=torch.bool)
        elif self.mask == 'train_val':
            mask = train_mask | val_mask
        elif self.mask == 'train_val_test':
            mask = train_mask | val_mask | test_mask
        elif self.mask == 'train_val_unlabeled':
            labeled_mask = train_mask | val_mask | test_mask
            unlabeled_mask = torch.ones_like(train_mask, dtype=torch.bool) ^ labeled_mask
            mask = train_mask | val_mask | unlabeled_mask
        else:
            raise Exception('The setting of `mask` is not supported')
        ce_loss = F.cross_entropy(outs['feats'][-1][train_mask], y[train_mask])

        if self.method == 'none':
            loss = ce_loss

        elif self.method == 'soft':
            kd_loss = self.soft_target_loss(outs['feats'][-1][mask], knowledge['feats'][-1][mask])
            loss = (1 - self.alpha) * ce_loss + self.alpha * kd_loss
            if self.verbose:
                print(f'ce_loss: {(1 - self.alpha) * ce_loss / loss : .2%}, kl_loss: {self.alpha * kd_loss / loss : .2%}')
        
        elif self.method == 'logit':
            kd_loss = self.logit_loss(outs['feats'][-1][mask], knowledge['feats'][-1][mask])
            loss = (1 - self.alpha) * ce_loss + self.alpha * kd_loss
            if self.verbose:
                print(f'ce_loss: {(1 - self.alpha) * ce_loss / loss : .2%}, kl_loss: {self.alpha * kd_loss / loss : .2%}')
            
        elif self.method == 'hidden':
            kd_loss = self.hidden_loss(outs['feats'][0][mask], knowledge['feats'][0][mask])
            loss = (1 - self.alpha) * ce_loss + self.alpha * kd_loss
            if self.verbose:
                print(f'ce_loss: {(1 - self.alpha) * ce_loss / loss : .2%}, kl_loss: {self.alpha * kd_loss / loss : .2%}')

        elif self.method == 'soft_link':
            label_loss = self.soft_target_loss(outs['feats'][-1][mask], knowledge['feats'][-1][mask])
            link_loss = self.link_predictor.link_loss(data.edge_index, outs['feats'][0])
            loss = (1 - self.alpha) * label_loss + self.alpha * link_loss
            if self.verbose:
                print(f'soft_loss: {(1 - self.alpha) * label_loss / loss : .2%}, link_loss: {self.alpha * link_loss / loss : .2%}')
        
        elif self.method == 'ce_link':
            link_loss = self.link_predictor.link_loss(data.edge_index, outs['feats'][0])
            loss = (1 - self.alpha) * ce_loss + self.alpha * link_loss
            if self.verbose:
                print(f'ce_loss: {(1 - self.alpha) * ce_loss / loss : .2%}, link_loss: {self.alpha * link_loss / loss : .2%}')
        return loss

    def soft_target_loss(self, out_s, out_t):
        return F.kl_div(F.log_softmax(out_s / self.T, dim=1), F.softmax(out_t / self.T, dim=1), reduction='batchmean') * (self.T * self.T)

    def logit_loss(self, out_s, out_t):
        return F.mse_loss(out_s, out_t)

    def hidden_loss(self, out_s, out_t):
        return F.mse_loss(out_s, out_t)


class LinkPredictor(torch.nn.Module):
    def __init__(self, num_hiddens, neg_k):
        super().__init__()
        self.num_hiddens = num_hiddens
        self.neg_k = neg_k
        self.predictor = nn.Sequential(
            nn.Linear(2*num_hiddens, num_hiddens), nn.ReLU(),
            nn.Linear(num_hiddens, 1)
        )

    def generate_samples(self, edge_index, x):
        num_nodes = x.shape[0]
        num_edges = edge_index.shape[1]
        neg_index = negative_sampling(edge_index, num_neg_samples=num_nodes * self.neg_k)
        pairs = torch.cat([edge_index, neg_index], dim=-1)
        labels = torch.cat([torch.ones(num_edges), torch.zeros(num_nodes * self.neg_k)], dim=-1).to(pairs.device)
        return pairs, labels

    def link_loss(self, pos_edges, x):
        pairs, labels = self.generate_samples(pos_edges, x)
        us, vs = pairs
        feats = torch.cat([x[us], x[vs]], dim=-1)
        preds = self.predictor(feats)
        loss = F.binary_cross_entropy_with_logits(preds.flatten(), labels)
        return loss


