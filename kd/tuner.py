import copy
import joblib
import numpy as np
import os.path as osp
import optuna
from tabulate import tabulate
import torch
from kd.experiment import Experiment
from kd.configs.config import load_config


class Tuner:
    def __init__(self, exp_cfg, tuner_cfg, dataset=None):
        self.exp_cfg = exp_cfg
        self.tuner_cfg = tuner_cfg
        self.dataset = dataset
        self.grid_search = self.tuner_cfg.get('grid_search', None)

    def parse_tune_space(self, trial, name, spcfg):
        '''
        suggest a choice for `name` param according to `spcfg` into the trial
        '''
        if spcfg['func'] == 'suggest_categorical':
            return trial.suggest_categorical(name, spcfg['choices'])
        elif spcfg['func'] == 'suggest_float':
            return trial.suggest_float(name, low=spcfg['low'], high=spcfg['high'], step=spcfg['step'], log=spcfg['log'])
        elif spcfg['func'] == 'suggest_int':
            return trial.suggest_int(name, low=spcfg['low'], high=spcfg['high'], step=spcfg['step'], log=spcfg['log'])
        
    def update_dict_by_keys(self, d, keys, value):
        td = d
        for k in keys[:-1]:
            td = td[k]
        td[keys[-1]] = value
        return d

    def suggestor(self, trial, exp_cfg, tuner_cfg):
        cfg = copy.deepcopy(exp_cfg)
        param_space = tuner_cfg.get('space', None)
        space_cfgs = tuner_cfg.get('space_cfgs', None)
        if param_space:
            for name, space in param_space.items():
                suggest_value = trial.suggest_categorical(name, space)
                self.update_dict_by_keys(cfg, name.split('.'), suggest_value)
        if space_cfgs:
            for name, spcfg in space_cfgs.items():
                suggest_value = self.parse_tune_space(trial, name, spcfg)
                self.update_dict_by_keys(cfg, name.split('.'), suggest_value)
        return cfg

    def experiment(self, cfg):
        exp = Experiment(cfg, dataset=self.dataset)
        res = []
        for _ in range(self.tuner_cfg.n_trial_runs):
            trainer = exp.run_single()
            train1, best_idx, train, valid, test = trainer.logger.report(verbose=False)
            res.append([train1, best_idx, train, valid, test])
            torch.cuda.empty_cache()
        res = np.array(res)
        return res

    def objective(self, trial):
        cfg = self.suggestor(trial, self.exp_cfg, self.tuner_cfg)
        res = self.experiment(cfg)
        test_acc = res[:, 4].mean()
        trial.set_user_attr("val_acc", res[:, 3].mean())
        trial.set_user_attr("val_std", res[:, 3].std())
        trial.set_user_attr("test_acc", res[:, 4].mean())
        trial.set_user_attr("test_std", res[:, 4].std())
        trial.set_user_attr("config", cfg)
        return test_acc

    def tune(self):
        if self.grid_search:
            sp = self.tuner_cfg.space
            study = optuna.create_study(direction='maximize', sampler=optuna.samplers.GridSampler(sp))
        else:
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
        select_columns = df.columns[df.columns.str.startswith('user_attrs') | df.columns.str.startswith('params')]
        df = df.sort_values('user_attrs_test_acc', ascending=False)[:5]
        df = df[select_columns].drop(columns=['user_attrs_config'])
        print(tabulate(df, headers=df.columns))
