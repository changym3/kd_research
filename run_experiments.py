import argparse
from copy import deepcopy
from kd.configs import load_config
from kd.Experiment import Experiment
from kd.configs.config import prepare_experiment_cfg



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # parser.add_argument('--model', type=str, default='MLP')
    # parser.add_argument('--dataset', type=str, default='Cora')
    # parser.add_argument('--gpu', type=int, default=1)
    
    parser.add_argument('--model_cfg', type=str, default='./examples/MLP.yaml')
    parser.add_argument('--dataset', type=str, default='Cora')

    parser.add_argument('--n_runs', type=int, default=1)

    args = parser.parse_args()

    model_cfg = load_config(args.model_cfg)
    exp_cfg= prepare_experiment_cfg(model_cfg, args.dataset)

    expt = Experiment(exp_cfg, n_runs=args.n_runs)
    expt.run()

    # experiment(args.model, args.dataset, gpu=args.gpu, cfg_path=args.cfg_path)


