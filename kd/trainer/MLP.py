import torch
from torch_geometric.nn.models import MLP

from kd.utils.evaluator import Evaluator
from kd.utils.logger import Logger

class MLPTrainer:
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
        loss = criterion(out[data.train_mask], data.y[data.train_mask].view(-1))
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