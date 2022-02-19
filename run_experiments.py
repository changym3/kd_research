import argparse
from kd.experiment import Experiment
from kd.configs import prepare_experiment_cfg, load_config


def update_cfg_by_args(cfg, args):
    cfg.trainer.ckpt_dir = args.ckpt_dir

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # parser.add_argument('--model', type=str, default='MLP')
    # parser.add_argument('--dataset', type=str, default='Cora')
    # parser.add_argument('--gpu', type=int, default=1)
    
    parser.add_argument('--model_cfg', type=str, default='./examples/MLP.yaml')
    parser.add_argument('--dataset', type=str, default='Cora')
    parser.add_argument('--n_runs', type=int, default=1)

    # update
    parser.add_argument('--ckpt_dir', type=str, default='./examples/ckpt/test_GAT')


    args = parser.parse_args()

    model_cfg = load_config(args.model_cfg)
    exp_cfg= prepare_experiment_cfg(model_cfg, args.dataset)

    expt = Experiment(exp_cfg, n_runs=args.n_runs)
    expt.run()

    # experiment(args.model, args.dataset, gpu=args.gpu, cfg_path=args.cfg_path)


