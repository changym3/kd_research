from collections import defaultdict
import os.path as osp
import time
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
    def __init__(self, cfg, data) -> None:
        super().__init__()
        self.cfg = cfg
        self.data = data
        self.af = AugmentedFeatures(cfg.model.aug_hop, cfg.model.get('aug_path', None))
        self.feat_list = self.af.augment_features(self.data)
        # self.mimic = cfg.model.mimic

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
        # if self.mimic:
        #     self.mimic_fc = MLP([num_hiddens, num_hiddens, num_hiddens],
        #                     dropout=dropout, batch_norm=batch_norm)
        # else:
        #     self.mimic_fc = nn.Identity()
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


class KDModelTrainer:
    def __init__(self, cfg, dataset, device):
        time_start=time.time()
        self.cfg = cfg
        self.dataset = dataset
        self.data = dataset[0].to(device)
        self.device = device
        self.kd_module = KDModule(cfg, verbose=cfg.trainer.verbose).to(device)
        self.knowledge = self.setup_knowledge(
            osp.join(cfg.trainer.kd.knowledge_dir, 'knowledge.pt'), device)
        self.model = KD_GAMLP(cfg, data=self.data).to(device)

        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=cfg.trainer.lr, weight_decay=cfg.trainer.weight_decay)
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
            loss = self.train_epoch(self.model, self.data, self.optimizer)
            time_end=time.time()
            train_time += (time_end - time_start)
            
            time_start=time.time()
            ks, train_acc, val_acc, test_acc = self.eval_epoch(
                self.evaluator, self.data, self.model)
            time_end=time.time()
            inference_time += (time_end - time_start)
            self.logger.add_result(
                epoch, loss, train_acc, val_acc, test_acc, verbose=self.cfg.trainer.verbose)
            if self.checkpoint:
                self.checkpoint.report(epoch, self.model, val_acc)
        if self.checkpoint:
            self.save_knowledge(self.model, self.data)
        
        print('Training Time:', train_time, 'seconds')
        print('Inference Time:', inference_time, 'seconds')
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
        loss = self.kd_loss(ks, data)
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

    def kd_loss(self, ks, data):
        loss = self.kd_module.loss(ks, self.knowledge, data)
        return loss

    def setup_knowledge(self, kno_path, device):
        knowledge = torch.load(kno_path)['knowledge']
        feats = [f.to(device) for f in knowledge['feats']]
        knowledge['feats'] = feats
        return knowledge


class KDModule(torch.nn.Module):
    def __init__(self, cfg, verbose=None) -> None:
        super().__init__()
        self.verbose = verbose
        self.cfg = cfg
        self.method = self.cfg.trainer.kd.method
        self.mask = self.cfg.trainer.kd.mask
        self.mimics_mask = self.cfg.trainer.kd.mimics_mask
        self.T = self.cfg.trainer.kd.temperature
        self.alpha = self.cfg.trainer.kd.get('alpha')
        self.beta = self.cfg.trainer.kd.get('beta')
    
    def get_mask(self, name, train_mask, val_mask, test_mask):
        assert len(train_mask) == len(val_mask) and len(
            val_mask) == len(test_mask)
        if name == 'all':
            mask = torch.ones_like(train_mask, dtype=torch.bool)
        elif name == 'train':
            mask = train_mask
        elif name == 'train_val':
            mask = train_mask | val_mask
        elif name == 'train_val_test':
            mask = train_mask | val_mask | test_mask
        elif name == 'train_val_unlabeled':
            labeled_mask = train_mask | val_mask | test_mask
            unlabeled_mask = torch.ones_like(
                train_mask, dtype=torch.bool) ^ labeled_mask
            mask = train_mask | val_mask | unlabeled_mask
        elif name == 'unlabeled':
            labeled_mask = train_mask | val_mask | test_mask
            mask = torch.ones_like(train_mask, dtype=torch.bool) ^ labeled_mask
        else:
            raise Exception('The setting of `mask` is not supported')
        assert mask.sum() > 0
        return mask

    def loss(self, ks, knowledge, data):
        y, train_mask, val_mask, test_mask = data.y, data.train_mask, data.val_mask, data.test_mask
        mask = self.get_mask(self.mask, train_mask, val_mask, test_mask)
        mimics_mask = self.get_mask(self.mimics_mask, train_mask, val_mask, test_mask)

        ce_loss = F.cross_entropy(
            ks['projection'][-1][train_mask], y[train_mask])

        if self.method == 'none':
            loss = ce_loss

        elif self.method == 'soft':            
            kd_loss = self.soft_target_loss(
                ks['projection'][-1][mask], knowledge['feats'][-1][mask])
            loss = (1 - self.alpha) * ce_loss + self.alpha * kd_loss
            if self.verbose:
                print(f'ce_loss: {ce_loss : .4f}), kd_loss: {kd_loss : .4f}')
                print(
                    f'ce_loss: {(1 - self.alpha) * ce_loss / loss : .2%}, kl_loss: {self.alpha * kd_loss / loss : .2%}')
            
        elif self.method == 'logit':
            kd_loss = F.mse_loss(ks['projection'][-1][mask], knowledge['feats'][-1][mask])
            loss = (1 - self.alpha) * ce_loss + self.alpha * kd_loss
            if self.verbose:
                print(f'ce_loss: {ce_loss : .4f}), kd_loss: {kd_loss : .4f}')
                print(
                    f'ce_loss: {(1 - self.alpha) * ce_loss / loss : .2%}, kl_loss: {self.alpha * kd_loss / loss : .2%}')

        elif self.method == 'feats':
            kd_loss = 0
            
            for i in range(2):
                kd_loss += F.mse_loss(ks['projection']
                                      [i][mask], knowledge['feats'][i][mask])
            kd_loss /= 2
            loss = (1 - self.alpha) * ce_loss + self.alpha * kd_loss
            if self.verbose:
                print(f'ce_loss: {ce_loss : .4f}), kd_loss: {kd_loss : .4f}')
                print(
                    f'ce_loss: {(1 - self.alpha) * ce_loss / loss : .2%}, kl_loss: {self.alpha * kd_loss / loss : .2%}')

        elif self.method == 'mimics':
            mimic_loss = 0
            for i in range(2):
                mimic_loss += F.mse_loss(ks['inceptions'][i+1]
                                         [mimics_mask], knowledge['feats'][i][mimics_mask])
            mimic_loss /= 2

            kd_loss = mimic_loss
            loss = (1 - self.alpha) * ce_loss + self.alpha * kd_loss
            if self.verbose:
                print(f'ce_loss: {ce_loss : .4f}), kd_loss: {kd_loss : .4f}')
                print(
                    f'ce_loss: {(1 - self.alpha) * ce_loss / loss : .2%}, kl_loss: {self.alpha * kd_loss / loss : .2%}')
                
        elif self.method == 'soft_mimics':
            mimic_loss = 0
            num_shallow = min(2, self.cfg.model.aug_hop)
            for i in range(num_shallow):
                mimic_loss += F.mse_loss(ks['inceptions'][i+1]
                                         [mimics_mask], knowledge['feats'][i][mimics_mask])
            mimic_loss /= num_shallow
            soft_loss = self.soft_target_loss(
                ks['projection'][-1][mask], knowledge['feats'][-1][mask])
            
            alpha = self.alpha
            beta = self.beta
            kd_loss = beta * soft_loss + (1-beta) * mimic_loss
            loss = (1-alpha) * ce_loss + alpha * kd_loss
            
            if self.verbose:
                print(f'ce_loss: {ce_loss : .4f}), soft_loss: {soft_loss : .4f}, mimic_loss: {mimic_loss : .4f}')
                print(f'kd_loss: {alpha * kd_loss / loss : .2%}, mimics_loss: {(1-beta) * mimic_loss / kd_loss : .2%}')

        return loss

    def soft_target_loss(self, out_s, out_t):
        return F.kl_div(F.log_softmax(out_s / self.T, dim=1), F.softmax(out_t / self.T, dim=1), reduction='batchmean') * (self.T * self.T)
