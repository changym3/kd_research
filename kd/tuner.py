import copy
import joblib
import numpy as np
import os.path as osp
import optuna
from tabulate import tabulate
from kd.experiment import Experiment
from kd.configs.config import load_config


class Tuner:
    def __init__(self, exp_cfg, tuner_cfg, dataset=None):
        self.exp_cfg = self.adjust_cfg(exp_cfg)
        self.tuner_cfg = tuner_cfg
        self.dataset = dataset

    def adjust_cfg(self, cfg):
        cfg = copy.deepcopy(cfg)
        cfg.trainer.ckpt_dir = None
        cfg.trainer.verbose = False
        return cfg

    def parse_tune_space(self, trial, name, space):
        '''
        suggest a choice for `name` param according to `space` into the trial
        '''
        if space['func'] == 'suggest_categorical':
            return trial.suggest_categorical(name, space['choices'])
        elif space['func'] == 'suggest_float':
            return trial.suggest_float(name, low=space['low'], high=space['high'], step=space['step'], log=space['log'])
        elif space['func'] == 'suggest_int':
            return trial.suggest_int(name, low=space['low'], high=space['high'], step=space['step'], log=space['log'])
    
    def update_dict_by_keys(self, d, keys, value):
        td = d
        for k in keys[:-1]:
            td = td[k]
        td[keys[-1]] = value
        return d

    def suggestor(self, trial, exp_cfg, space_cfg):
        cfg = copy.deepcopy(exp_cfg)
        for name, space in space_cfg.items():
            suggest_value = self.parse_tune_space(trial, name, space)
            self.update_dict_by_keys(cfg, name.split('.'), suggest_value)
        return cfg

    def experiment(self, cfg):
        exp = Experiment(cfg, dataset=self.dataset)
        res = []
        for _ in range(self.tuner_cfg.n_trial_runs):
            trainer = exp.run_single()
            train1, best_idx, train, valid, test = trainer.logger.report(verbose=False)
            res.append([train1, best_idx, train, valid, test])
        res = np.array(res)
        return res

    def objective(self, trial):
        cfg = self.suggestor(trial, self.exp_cfg, self.tuner_cfg.space)
        res = self.experiment(cfg)
        test_acc = res[:, 4].mean()
        trial.set_user_attr("val_acc", res[:, 3].mean())
        trial.set_user_attr("val_std", res[:, 3].std())
        trial.set_user_attr("test_acc", res[:, 4].mean())
        trial.set_user_attr("test_std", res[:, 4].std())
        trial.set_user_attr("config", cfg)
        return test_acc

    def tune(self):
        study = optuna.create_study(direction='maximize')
        study.optimize(self.objective, n_trials=self.tuner_cfg.n_trials)
        self.study = study

    def get_study(self):
        if hasattr(self, 'study'):
            return self.study
        else:
            raise Exception('No study in Tuner object')
    
    def save_study(self, study_path=None):
        if study_path is None:
            study_path = osp.join(self.tuner_cfg.study_dir, self.tuner_cfg.version)
        study = self.get_study()
        joblib.dump(study, study_path)
        print(f'Saved study into {study_path}.')
    
    def print_study_analysis(self, study=None):
        if study is None:
            study = self.get_study()
        df = study.trials_dataframe()
        select_columns = df.columns[df.columns.str.startswith('user_attrs') | df.columns.str.startswith('params_trainer')]
        df = df.sort_values('user_attrs_test_acc', ascending=False)[:5]
        df = df[select_columns].drop(columns=['user_attrs_config'])
        print(tabulate(df, headers=df.columns))
