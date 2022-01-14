from distutils.command.config import config
import torch


class Checkpoint():
    def __init__(self, config, ckpt_path, mode='max'):
        self.config = config
        self.ckpt_path = ckpt_path
        self.best_score = None
        self.best_iter = None
        self.iter_cnt = 0
        self.mode = mode

    def report(self, model, score):
        self.iter_cnt += 1
        if self.best_score is None:
            self.save_and_update(model, score)
        else:
            if self.mode == 'max' and score > self.best_score:
                self.save_and_update(model, score)
            elif self.mode == 'min' and score < self.best_score:
                self.save_and_update(model, score)            

    def save_and_update(self, model, score):
        self.best_score = score
        self.best_iter = self.iter_cnt
        self.save(model)
    
    def save(self, model):
        torch.save({
            'model_state_dict': model.state_dict(),
            'config': self.config
            }, self.ckpt_path)


