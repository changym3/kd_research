import copy
import joblib
import numpy as np
import optuna
from kd.experiment import Experiment
from kd.configs.config import load_config

class Tuner:
    def __init__(self, exp_cfg, tuner_cfg, n_trials, n_repeats=1, dataset=None):
        self.cfg = self.adjust_cfg(exp_cfg)
        self.tuner_cfg = tuner_cfg
        self.n_trials = n_trials
        self.n_repeats = n_repeats
        self.dataset = dataset

    def adjust_cfg(self, cfg):
        cfg = copy.deepcopy(cfg)
        cfg.trainer.ckpt_dir = None
        cfg.trainer.verbose = False
        return cfg

    def parse_tune_info(self, trial, name, info):
        '''
        suggest a choice for `name` param according to `info` into the trial
        '''
        if info['func'] == 'suggest_categorical':
            return trial.suggest_categorical(name, info['choices'])
        elif info['func'] == 'suggest_float':
            return trial.suggest_float(name, low=info['low'], high=info['high'], step=info['step'], log=info['log'])
        elif info['func'] == 'suggest_int':
            return trial.suggest_int(name, low=info['low'], high=info['high'], step=info['step'], log=info['log'])
    
    def update_dict_by_keys(self, d, keys, value):
        td = d
        for k in keys[:-1]:
            td = td[k]
        td[keys[-1]] = value
        return d

    def suggestor(self, trial, param_cfg, tuner_cfg):
        cfg = copy.deepcopy(param_cfg)
        for name, info in tuner_cfg.items():
            suggest_value = self.parse_tune_info(trial, name, info)
            self.update_dict_by_keys(cfg, name.split('.'), suggest_value)
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
        cfg = self.suggestor(trial, self.cfg, self.tuner_cfg)
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