import argparse
import os.path as osp
from kd.configs import load_config, prepare_experiment_cfg
from kd.data.dataset import build_dataset
from kd.tuner import Tuner


def update_cfg_by_args(cfg, args):
    cfg.n_trials = args.n_trials
    cfg.version = args.version


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # parser.add_argument('--model', type=str, default='MLP')
    # parser.add_argument('--dataset', type=str, default='Cora')
    # parser.add_argument('--gpu', type=int, default=1)
    
    parser.add_argument('--model_cfg', type=str, default='./examples/KDMLP.yaml')
    parser.add_argument('--tuner_cfg', type=str, default='./examples/KDMLP_tuner.yaml')
    parser.add_argument('--dataset', type=str, default='Cora')
    # update
    parser.add_argument('--n_trials', type=int, default=5)
    parser.add_argument('--version', type=str, default='KDMLP.study')
    args = parser.parse_args()

    model_cfg = load_config(args.model_cfg)
    tuner_cfg = load_config(args.tuner_cfg)
    exp_cfg= prepare_experiment_cfg(model_cfg, args.dataset)
    update_cfg_by_args(tuner_cfg, args)

    dataset = build_dataset(args.dataset)
    tuner = Tuner(exp_cfg, tuner_cfg, dataset=dataset)
    tuner.tune()
    tuner.save_study()

# python run_tuner.py --model_cfg examples/KDMLP.yaml --tuner_cfg examples/KDMLP_tuner.yaml --dataset Cora --n_trials 1000