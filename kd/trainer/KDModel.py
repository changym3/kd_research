from aifc import Error
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


class KD_GAMLP(nn.Module):
    def __init__(self, cfg, data, knowledge) -> None:
        super().__init__()
        self.cfg = cfg
        self.data = data
        self.af = AugmentedFeatures(cfg)
        self.knowledge = knowledge
        self.feats = self.af.augment_features(self.data, self.knowledge)
        
        self.num_channels = cfg.model.knn.hop + cfg.model.raw.hop + 1
        self.combine_type = cfg.model.feat_combine
        self.channel_combine = ChannelCombine(cfg.dataset.num_features , self.combine_type, self.num_channels)
        self.student_model = self.build_student_model(cfg)

    def build_student_model(self, cfg):
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
    
    # def forward(self, model, data):
    #     x = self.channel_combine(self.feats)
    #     if self.cfg.meta.student_name == 'MLP':
    #         out = model(x)
    #     else:
    #         out = model(x, data.edge_index)
    #     return out
    
    def forward_state(self, data):
        x = self.channel_combine(self.feats)
        outs = K.get_model_state(self.student_model, data, self.cfg.meta.student_name, x=x)
        return outs
    
    
class KDModelTrainer:
    def __init__(self, cfg, dataset, device):
        self.cfg = cfg
        self.dataset = dataset
        self.data = dataset[0].to(device)
        self.device = device
        self.kd_module = KDModule(cfg, verbose=cfg.trainer.verbose).to(device)
        self.knowledge = self.setup_knowledge(osp.join(cfg.trainer.kd.knowledge_dir, 'knowledge.pt'), device)
        self.model = KD_GAMLP(cfg, data=self.data, knowledge=self.knowledge).to(device)

        self.evaluator = Evaluator()
        self.logger = Logger()
        if self.cfg.trainer.get('ckpt_dir', None) is None:
            self.checkpoint = None
        else:
            self.checkpoint = Checkpoint(cfg, cfg.trainer.ckpt_dir)
        self.optimizer = torch.optim.Adam([
            {'params': self.model.parameters()},
            {'params': self.kd_module.parameters()}
            ], lr=cfg.trainer.lr, weight_decay=cfg.trainer.weight_decay)

    def fit(self):
        for epoch in range(self.cfg.trainer.epochs):
            loss = self.train_epoch(self.model, self.data, self.optimizer)
            outs, train_acc, val_acc, test_acc = self.eval_epoch(self.evaluator, self.data, self.model)
            self.logger.add_result(epoch, loss, train_acc, val_acc, test_acc, verbose=self.cfg.trainer.verbose)
            if self.checkpoint:
                self.checkpoint.report(epoch, self.model, val_acc)
        if self.checkpoint:
            self.save_knowledge(self.model, self.data)

    def save_knowledge(self, model, data):
        ckpt_dir = self.cfg.trainer.ckpt_dir
        model.load_state_dict(torch.load(osp.join(ckpt_dir, 'model.pt'))['model_state_dict'])
        outs, train_acc, val_acc, test_acc = self.eval_epoch(self.evaluator, data, model)

        kno_path = osp.join(ckpt_dir, 'knowledge.pt')
        torch.save({'knowledge': outs}, kno_path)
        print(f'Save predictions into file {kno_path}.')
        print(f'The Prediction, Train: {train_acc:.4f}, Val: {val_acc:.4f}, Test: {test_acc:.4f}')

    def train_epoch(self, model, data, optimizer):
        model.train()
        optimizer.zero_grad()
        outs = model.forward_state(data)
        loss = self.kd_loss(outs, data)
        loss.backward()
        optimizer.step()
        return float(loss)

    @torch.no_grad()
    def eval_epoch(self, evaluator, data, model):
        model.eval()
        outs = model.forward_state(data)
        out = outs['feats'][-1]
        train_acc = evaluator.eval(out[data.train_mask], data.y[data.train_mask])['acc']
        val_acc = evaluator.eval(out[data.val_mask], data.y[data.val_mask])['acc']
        test_acc = evaluator.eval(out[data.test_mask], data.y[data.test_mask])['acc']
        return outs, train_acc, val_acc, test_acc

    def kd_loss(self, outs, data):
        loss = self.kd_module.loss(outs, self.knowledge, data)
        return loss

    def setup_knowledge(self, kno_path, device):
        knowledge = torch.load(kno_path)['knowledge'].to(device)
        return knowledge

class KDModule(torch.nn.Module):
    def __init__(self, total_cfg, verbose=None) -> None:
        super().__init__()
        self.verbose = verbose
        self.total_cfg = total_cfg
        self.cfg = total_cfg.trainer.kd

        self.method = self.cfg.method
        self.mask = self.cfg.mask
        self.T = self.cfg.temperature
        self.alpha = self.cfg.get('alpha')
        self.beta = self.cfg.get('beta')

        num_hiddens = self.total_cfg.model.num_hiddens
        self.link_predictor = LinkPredictor(num_hiddens, self.cfg.get('neg_k'))
        self.decoder = nn.Sequential(nn.Linear(num_hiddens, num_hiddens))

    def loss(self, outs, knowledge, data):
        y, train_mask, val_mask, test_mask = data.y, data.train_mask, data.val_mask, data.test_mask
        assert len(train_mask) == len(val_mask) and len(val_mask) == len(test_mask)
        if self.mask == 'all':
            mask = torch.ones_like(train_mask, dtype=torch.bool)
        elif self.mask == 'train':
            mask = train_mask
        elif self.mask == 'train_val':
            mask = train_mask | val_mask
        elif self.mask == 'train_val_test':
            mask = train_mask | val_mask | test_mask
        elif self.mask == 'train_val_unlabeled':
            labeled_mask = train_mask | val_mask | test_mask
            unlabeled_mask = torch.ones_like(train_mask, dtype=torch.bool) ^ labeled_mask
            mask = train_mask | val_mask | unlabeled_mask
        elif self.mask == 'unlabeled':
            labeled_mask = train_mask | val_mask | test_mask
            mask = torch.ones_like(train_mask, dtype=torch.bool) ^ labeled_mask
        else:
            raise Exception('The setting of `mask` is not supported')
        ce_loss = F.cross_entropy(outs['feats'][-1][train_mask], y[train_mask])

        if self.method == 'none':
            loss = ce_loss

        elif self.method == 'soft':
            kd_loss = self.soft_target_loss(outs['feats'][-1][mask], knowledge['feats'][-1][mask])
            loss = (1 - self.alpha) * ce_loss + self.alpha * kd_loss
            if self.verbose:
                print(f'ce_loss: {ce_loss : .4f}), kd_loss: {kd_loss : .4f}')
                print(f'ce_loss: {(1 - self.alpha) * ce_loss / loss : .2%}, kl_loss: {self.alpha * kd_loss / loss : .2%}')
        
        elif self.method == 'logit':
            kd_loss = self.logit_loss(outs['feats'][-1][mask], knowledge['feats'][-1][mask])
            loss = (1 - self.alpha) * ce_loss + self.alpha * kd_loss
            if self.verbose:
                print(f'ce_loss: {ce_loss : .4f}), kd_loss: {kd_loss : .4f}')
                print(f'ce_loss: {(1 - self.alpha) * ce_loss / loss : .2%}, kl_loss: {self.alpha * kd_loss / loss : .2%}')
            
        elif self.method == 'hidden':
            kd_loss = self.hidden_loss(outs['feats'][0][mask], knowledge['feats'][0][mask])
            loss = (1 - self.alpha) * ce_loss + self.alpha * kd_loss
            if self.verbose:
                print(f'ce_loss: {ce_loss : .4f}), kd_loss: {kd_loss : .4f}')
                print(f'ce_loss: {(1 - self.alpha) * ce_loss / loss : .2%}, kl_loss: {self.alpha * kd_loss / loss : .2%}')
        
        elif self.method == 'feats':
            kd_loss = F.mse_loss(outs['feats'][0][mask], knowledge['feats'][0][mask]) + F.mse_loss(outs['feats'][-1][mask], knowledge['feats'][-1][mask])
            kd_loss /= 2
            loss = (1 - self.alpha) * ce_loss + self.alpha * kd_loss
            if self.verbose:
                print(f'ce_loss: {ce_loss : .4f}), kd_loss: {kd_loss : .4f}')
                print(f'ce_loss: {(1 - self.alpha) * ce_loss / loss : .2%}, kl_loss: {self.alpha * kd_loss / loss : .2%}')

        elif self.method == 'soft_link':
            label_loss = self.soft_target_loss(outs['feats'][-1][mask], knowledge['feats'][-1][mask])
            link_loss = self.link_predictor.link_loss(data.edge_index, outs['feats'][0])
            loss = (1 - self.alpha) * label_loss + self.alpha * link_loss
            if self.verbose:
                print(f'soft_loss: {(1 - self.alpha) * label_loss / loss : .2%}, link_loss: {self.alpha * link_loss / loss : .2%}')
        
        elif self.method == 'ce_link':
            kd_loss = self.link_predictor.link_loss(data.edge_index, outs['feats'][0])
            loss = (1 - self.alpha) * ce_loss + self.alpha * kd_loss
            if self.verbose:
                print(f'ce_loss: {ce_loss : .4f}), kd_loss: {kd_loss : .4f}')
                print(f'ce_loss: {(1 - self.alpha) * ce_loss / loss : .2%}, kd_loss: {self.alpha * kd_loss / loss : .2%}')
        
        elif self.method == 'ce_recover':
            # input = outs['feats'][0][mask] # encoder is the first layer of MLP
            hidden = outs['feats'][1][mask]
            output_truth = knowledge['feats'][0][mask]
            kd_loss = F.mse_loss(output_truth, self.decoder(hidden))
            loss = (1 - self.alpha) * ce_loss + self.alpha * kd_loss
            if self.verbose:
                print(f'ce_loss: {ce_loss : .4f}), kd_loss: {kd_loss : .4f}')
                print(f'ce_loss: {(1 - self.alpha) * ce_loss / loss : .2%}, kd_loss: {self.alpha * kd_loss / loss : .2%}')
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


