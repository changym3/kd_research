from inspect import modulesbyfile
from kd.evaluator import Evaluator
import torch
from torch_geometric.nn.models import MLP

class MLPTrainer:
    def __init__(self, config, dataset, device):
        self.config = config
        self.dataset = dataset
        self.data = dataset[0].to(device)
        self.device = device
        self.model = self.build_model(config).to(device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=config.trainer.lr, weight_decay=config.trainer.weight_decay)
        self.criterion = torch.nn.CrossEntropyLoss()
        self.evaluator = Evaluator()

    def build_model(self, config):
        num_features = config.dataset.num_features
        num_hiddens = config.model.num_hiddens
        num_layers = config.model.num_layers
        num_classes = config.dataset.num_classes
        dropout = config.model.dropout
        batch_norm = config.model.batch_norm

        channel_list = [num_features, *([num_hiddens] * (num_layers - 1)), num_classes]
        model = MLP(channel_list, dropout, batch_norm=batch_norm)
        return model

    def fit(self):
        for epoch in range(1, self.config.trainer.epochs):
            loss = self.train_epoch(self.model, self.data, self.optimizer, self.criterion)
            train_acc, val_acc, test_acc = self.eval_epoch(self.evaluator, self.data, model=self.model)
            print(f'Epoch {epoch:4d}, Loss: {loss:.4f}, Train: {train_acc:.4f}, Val: {val_acc:.4f}, Test: {test_acc:.4f}')
            
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