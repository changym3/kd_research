import torch
from torch_geometric.nn.models import MLP
import torch_geometric.transforms as T
from kd.utils.evaluator import Evaluator
from kd.utils.logger import Logger

class MLPTrainer:
    def __init__(self, cfg, dataset, device):
        self.cfg = cfg
        self.dataset = dataset
        self.data = self.augment_features(dataset[0].to(device))
        self.device = device
        self.model = MLPTrainer.build_model(cfg).to(device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=cfg.trainer.lr, weight_decay=cfg.trainer.weight_decay)
        self.criterion = torch.nn.CrossEntropyLoss()
        self.evaluator = Evaluator()
        self.logger = Logger()
    
    def augment_features(self, data):
        aug_k = self.cfg.model.aug.k
        aug_combine = self.cfg.model.aug.combine
        transform = T.SIGN(aug_k)
        data = transform(data)
        x_list = [data.x]
        for i in range(1, aug_k+1):
            x_list.append(getattr(data, f'x{i}'))
            delattr(data, f'x{i}')
        if aug_combine == 'cat':
            data.x = torch.cat(x_list, dim=-1)
            self.cfg.dataset.num_features = self.cfg.dataset.num_features * (aug_k + 1)
        return data

    def build_model(cfg):
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