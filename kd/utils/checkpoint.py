from distutils.command.config import config
import os
import shutil
import torch


class Checkpoint():
    def __init__(self, config, ckpt_dir, mode='max'):
        self.config = config
        self.ckpt_dir = ckpt_dir
        self.best_score = None
        self.best_epoch = None
        self.mode = mode
        
        self.prepare_dir(ckpt_dir)

    def prepare_dir(self, ckpt_dir):
        if os.path.exists(ckpt_dir):
            shutil.rmtree(ckpt_dir)
        os.makedirs(ckpt_dir)

    def report(self, epoch, model, score):
        if self.best_score is None:
            self.save_and_update(epoch, model, score)
        else:
            if self.mode == 'max' and score > self.best_score:
                self.save_and_update(epoch, model, score)
            elif self.mode == 'min' and score < self.best_score:
                self.save_and_update(epoch, model, score)            

    def save_and_update(self, epoch, model, score):
        self.best_score = score
        self.best_epoch = epoch
        self.save(model)
    
    def save(self, model):
        ckpt_path = os.path.join(self.ckpt_dir, 'model.pt')
        cfg_path = os.path.join(self.ckpt_dir, 'config.pt')
        torch.save({'model_state_dict': model.state_dict()}, ckpt_path)
        torch.save({'config': self.config}, cfg_path)

