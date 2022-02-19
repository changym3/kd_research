import argparse
from kd.configs import load_config, prepare_experiment_cfg
from kd.data.dataset import build_dataset
from kd.tuner import Tuner


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # parser.add_argument('--model', type=str, default='MLP')
    # parser.add_argument('--dataset', type=str, default='Cora')
    # parser.add_argument('--gpu', type=int, default=1)
    
    parser.add_argument('--model_cfg', type=str, default='./examples/KDMLP.yaml')
    parser.add_argument('--tuner_cfg', type=str, default='./examples/KDMLP_tuner.yaml')
    parser.add_argument('--dataset', type=str, default='Cora')
    parser.add_argument('--n_trials', type=int, default=50)
    parser.add_argument('--n_repeats', type=int, default=1)
    parser.add_argument('--study_path', type=str, default='./examples/study/KDMLP.study')
    args = parser.parse_args()

    model_cfg = load_config(args.model_cfg)
    tuner_cfg = load_config(args.tuner_cfg)
    exp_cfg= prepare_experiment_cfg(model_cfg, args.dataset)

    dataset = build_dataset(exp_cfg.meta.dataset_name)
    tuner = Tuner(exp_cfg, tuner_cfg, args.n_trials, n_repeats=args.n_repeats, dataset=dataset)
    tuner.tune()
    tuner.save_study(args.study_path)

# python run_tuner.py --model_cfg examples/KDMLP.yaml --tuner_cfg examples/KDMLP_tuner.yaml --dataset Cora --n_trials 1000 --n_repeats 1