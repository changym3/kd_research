import argparse
from kd.experiments import experiment

if __name__ == '__main__':
    # run_experiment('GAT', 'Cora', gpu=1)
    # run_experiment('MLP', 'Cora', gpu=1)


    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='MLP')
    parser.add_argument('--dataset', type=str, default='Cora')
    parser.add_argument('--gpu', type=int, default=1)

    args = parser.parse_args()
    experiment(args.model, args.dataset, gpu=args.gpu)