import sys

sys.path.append('.')

import argparse
import copy
from kd.configs import load_config, prepare_experiment_cfg
from kd.data.dataset import build_dataset
from kd.Tuner import Tuner


def suggestor(trial, config):
    config = copy.deepcopy(config)
    config.trainer.kd.method = trial.suggest_categorical('method', [ 'soft', 'hidden', 'logit']) # ['none', 'soft', 'hidden', 'logit']
    # config.trainer.kd.mask = trial.suggest_categorical('mask', ['all']) # ['train_val', 'train_val_test', 'all']
    config.trainer.kd.alpha = trial.suggest_float('alpha', 0, 1)
    # config.trainer.kd.beta = trial.suggest_float('beta', 0, 1)
    # config.trainer.kd.gamma = trial.suggest_float('gamma', 0, 1)
    return config



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # parser.add_argument('--model', type=str, default='MLP')
    # parser.add_argument('--dataset', type=str, default='Cora')
    # parser.add_argument('--gpu', type=int, default=1)
    
    parser.add_argument('--model_cfg', type=str, default='./examples/KDMLP.yaml')
    parser.add_argument('--dataset', type=str, default='Cora')
    parser.add_argument('--n_trials', type=int, default=50)
    parser.add_argument('--n_repeats', type=int, default=1)
    parser.add_argument('--study_path', type=str, default='./examples/study/KDMLP.study')
    args = parser.parse_args()

    model_cfg = load_config(args.model_cfg)
    dataset_cfg = load_config('./examples/dataset_config.yaml')[args.dataset]
    exp_cfg= prepare_experiment_cfg(model_cfg, dataset_cfg)

    dataset = build_dataset(exp_cfg.meta.dataset_name)
    tuner = Tuner(exp_cfg, suggestor, args.n_trials, n_repeats=args.n_repeats, dataset=dataset)
    tuner.tune()
    tuner.save_study(args.study_path)

# python examples/KDMLP_tuner.py --model_cfg examples/KDMLP.yaml --dataset Cora --n_trials 1000 --n_repeats 3