import torch


class Logger:
    def __init__(self):
        self.results = []
    
    def add_result(self, epoch, loss, train_res, val_res, test_res, verbose=True):
        self.results.append([epoch, loss, train_res, val_res, test_res])
        if verbose:
            print(f'Epoch {epoch:4d}, Loss: {loss:.4f}, Train: {train_res:.4f}, Val: {val_res:.4f}, Test: {test_res:.4f}')
    
    def report(self, verbose=True):
        r = torch.as_tensor(self.results)
        train1 = r[:, 2].max().item()
        best_idx = r[:, 3].argmax()
        train = r[best_idx, 2].item()
        valid = r[best_idx, 3].item()
        test = r[best_idx, 4].item()
        if verbose:
            print(f'Highest Train: {train1}')
            print(f'Best Epoch : {best_idx}')
            print(f'Best Epoch - Train: {train}')
            print(f'Best Epoch - Valid: {valid}')
            print(f'Best Epoch - Test: {test}')
        return train1, best_idx, train, valid, test
