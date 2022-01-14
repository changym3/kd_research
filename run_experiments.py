import argparse
from kd.Experiment import Experiment

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # parser.add_argument('--model', type=str, default='MLP')
    # parser.add_argument('--dataset', type=str, default='Cora')
    # parser.add_argument('--gpu', type=int, default=1)

    parser.add_argument('--cfg_path', type=str, default='./examples/example_config.yaml')

    args = parser.parse_args()
    expt = Experiment(cfg_path=args.cfg_path)
    expt.run()

    # experiment(args.model, args.dataset, gpu=args.gpu, cfg_path=args.cfg_path)