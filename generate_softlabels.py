import argparse
import os
from easydict import EasyDict
import yaml

import torch

from kd.trainer import BasicGNNTrainer
from kd.data import build_dataset
from kd.utils.evaluator import Evaluator

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--ckpt_dir', type=str, default='./ckpt/test_GAT/')
    args = parser.parse_args()

    ckpt_path = os.path.join(args.ckpt_dir, 'model.pt')
    cfg_path = os.path.join(args.ckpt_dir, 'config.pt')  
    model_state_dict = torch.load(ckpt_path)['model_state_dict']
    config = torch.load(cfg_path)['config']
    model = BasicGNNTrainer.build_model(config)
    model.load_state_dict(model_state_dict)

    dataset = build_dataset(config.meta.dataset_name)
    data = dataset[0]

    with torch.no_grad():
        model.eval()
        y_pred = model(data.x, data.edge_index)        

    evaluator = Evaluator()
    pred_path = os.path.realpath(os.path.join(args.ckpt_dir, 'pred.pt'))
    torch.save({'pred': y_pred}, pred_path)
    print(f'Save predictions into file {pred_path}.')
    
    pred_acc = evaluator.eval(y_pred, data.y)['acc']
    print(f'The predictions have the accuracy of {pred_acc:.4f}.')

# python generate_softlabels.py --ckpt_dir ./ckpt/test_GAT