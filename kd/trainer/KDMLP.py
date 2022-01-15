import torch
import torch.nn.functional as F
from torch_geometric.nn.models import MLP

from kd.utils.evaluator import Evaluator
from kd.utils.logger import Logger


class KDMLPTrainer:
    def __init__(self, cfg, dataset, device):
        self.cfg = cfg
        self.dataset = dataset
        self.data = dataset[0].to(device)
        self.device = device
        self.model = self.build_model(cfg).to(device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=cfg.trainer.lr, weight_decay=cfg.trainer.weight_decay)
        self.criterion = torch.nn.CrossEntropyLoss()
        self.evaluator = Evaluator()
        self.logger = Logger()

        self.kd_cfg = cfg.trainer.kd
        self.soft_label = self.setup_softlabel(self.kd_cfg.softlabel_path, self.device)
        self.kd_module = KDModule(self.kd_cfg.temperature, self.kd_cfg.alpha)


    def build_model(self, cfg):
        num_features = cfg.dataset.num_features
        num_hiddens = cfg.model.num_hiddens
        num_layers = cfg.model.num_layers
        num_classes = cfg.dataset.num_classes
        dropout = cfg.model.dropout
        batch_norm = cfg.model.batch_norm

        channel_list = [num_features, *([num_hiddens] * (num_layers - 1)), num_classes]
        model = MLP(channel_list, dropout, batch_norm=batch_norm)
        return model

    def fit(self):
        for epoch in range(self.cfg.trainer.epochs):
            loss = self.train_epoch(self.model, self.data, self.optimizer, self.criterion)
            train_acc, val_acc, test_acc = self.eval_epoch(self.evaluator, self.data, model=self.model)
            self.logger.add_result(epoch, loss, train_acc, val_acc, test_acc, verbose=self.cfg.trainer.verbose)
            
    def train_epoch(self, model, data, optimizer, criterion):
        model.train()
        optimizer.zero_grad()
        out = model(data.x)
        # loss = self.kd_module.loss(out[data.train_mask], self.soft_label[data.train_mask], data.y[data.train_mask])
        loss = self.kd_module.loss(out, self.soft_label, data.y, data.train_mask, data.val_mask)
        # loss = criterion(out[data.train_mask], data.y[data.train_mask].view(-1))
        loss.backward()
        optimizer.step()
        return float(loss)

    #torch.no_grad()
    def eval_epoch(self, evaluator, data, out=None, model=None):
        assert out is not None or model is not None
        if out is None:
            model.eval()
            out = model(data.x)
        train_acc = evaluator.eval(out[data.train_mask], data.y[data.train_mask])['acc']
        val_acc = evaluator.eval(out[data.val_mask], data.y[data.val_mask])['acc']
        test_acc = evaluator.eval(out[data.test_mask], data.y[data.test_mask])['acc']
        return train_acc, val_acc, test_acc

    
    def setup_softlabel(self, pred_path, device):
        soft_label = torch.load(pred_path)['pred'].to(device)
        return soft_label



class KDModule:
    def __init__(self, temperature, alpha) -> None:
        self.T = temperature
        self.alpha = alpha

    def loss(self, logits, soft_y, y, train_mask, val_mask):
        tv_mask = torch.logical_or(train_mask, val_mask)
        kl_loss = self.get_kl_loss(logits[tv_mask], soft_y[tv_mask])
        ce_loss = F.cross_entropy(logits[train_mask], y[train_mask])
        loss = (1 - self.alpha) * ce_loss + self.alpha * kl_loss
        print(f'loss = {loss:.4f}, ce_loss = {ce_loss:.4f}, kl_loss = {kl_loss:.4f}')
        return loss

    def get_kl_loss(self, logits, soft_y):
        return F.kl_div(F.log_softmax(logits / self.T, dim=1), F.softmax(soft_y / self.T, dim=1)) * (self.T * self.T)

    def get_ce_loss(self, logits, y):
        return F.cross_entropy(logits, y)