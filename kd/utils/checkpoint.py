from distutils.command.config import config
import os
import torch


class Checkpoint():
    def __init__(self, config, ckpt_dir, mode='max'):
        self.config = config
        self.ckpt_dir = ckpt_dir
        self.best_score = None
        self.best_iter = None
        self.iter_cnt = 0
        self.mode = mode
        
        self.prepare_dir(ckpt_dir)

    def prepare_dir(self, ckpt_dir):
        if not os.path.exists(ckpt_dir):
            os.makedirs(ckpt_dir)

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
        ckpt_path = os.path.join(self.ckpt_dir, 'model.pt')
        cfg_path = os.path.join(self.ckpt_dir, 'config.pt')
        torch.save({'model_state_dict': model.state_dict()}, ckpt_path)
        torch.save({'config': self.config}, cfg_path)

