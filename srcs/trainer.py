from kd.srcs.evaluator import Evaluator
import torch

class Trainer:
    def __init__(self, config, model, dataset, device):
        self.config = config
        self.model = model.to(device)
        self.dataset = dataset
        self.data = dataset[0].to(device)
        self.device = device
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=config.trainer.lr, weight_decay=config.trainer.weight_decay)
        self.criterion = torch.nn.CrossEntropyLoss()
        self.evaluator = Evaluator()

    def fit(self):
        for epoch in range(1, self.config.trainer.epochs):
            loss = self.train_epoch(self.model, self.data, self.optimizer, self.criterion)
            train_acc, val_acc, test_acc = self.eval_epoch(self.evaluator, self.data, model=self.model)
            print(f'Epoch {epoch:4d}, Loss: {loss:.4f}, Train: {train_acc:.4f}, Val: {val_acc:.4f}, Test: {test_acc:.4f}')
            
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

