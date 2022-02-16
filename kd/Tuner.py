import copy
import joblib
import numpy as np
import optuna
from kd.Experiment import Experiment

class Tuner:
    def __init__(self, exp_cfg, suggestor, n_trials, n_repeats=1, dataset=None):
        self.cfg = self.adjust_cfg(exp_cfg)
        self.sugggestor = suggestor
        self.n_trials = n_trials
        self.n_repeats = n_repeats
        self.dataset = dataset

    def adjust_cfg(self, cfg):
        cfg = copy.deepcopy(cfg)
        cfg.trainer.ckpt_dir = None
        cfg.trainer.verbose = False
        return cfg
    
    def experiment(self, cfg):
        exp = Experiment(cfg, dataset=self.dataset)
        res = []
        for _ in range(self.n_repeats):
            trainer = exp.run_single()
            train1, best_idx, train, valid, test = trainer.logger.report()
            res.append([train1, best_idx, train, valid, test])
        res = np.array(res)
        return res

    def objective(self, trial):
        cfg = self.sugggestor(trial, self.cfg)
        res = self.experiment(cfg)
        val_acc = res[:, 3].mean()
        trial.set_user_attr("val_acc", val_acc)
        trial.set_user_attr("val_std", res[:, 3].std())
        trial.set_user_attr("test_acc", res[:, 4].mean())
        trial.set_user_attr("test_std", res[:, 4].std())
        return val_acc

    def tune(self):
        study = optuna.create_study(direction='maximize')
        study.optimize(self.objective, n_trials=self.n_trials)
        self.study = study

    def get_study(self):
        if hasattr(self, 'study'):
            return self.study
        else:
            raise Exception('No study in Tuner object')
    
    def save_study(self, study_path):
        study = self.get_study()
        joblib.dump(study, study_path)