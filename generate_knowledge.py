import argparse
import os
from easydict import EasyDict
import yaml

import torch

from kd.trainer import BasicGNNTrainer
from kd.data import build_dataset
from kd.utils.evaluator import Evaluator
from kd.utils.knowledge import get_knowledge

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--ckpt_dir', type=str, default='./ckpt/test_GAT/')
    parser.add_argument('--gpu', type=int, default=0)
    args = parser.parse_args()

    ckpt_path = os.path.join(args.ckpt_dir, 'model.pt')
    cfg_path = os.path.join(args.ckpt_dir, 'config.pt')  
    model_state_dict = torch.load(ckpt_path)['model_state_dict']
    config = torch.load(cfg_path)['config']
    model = BasicGNNTrainer.build_model(config)
    model.load_state_dict(model_state_dict)

    dataset = build_dataset(config.meta.dataset_name)
    device = torch.device(f'cuda:{args.gpu}')
    knowledge = get_knowledge(model, dataset, device)

    kno_path = os.path.realpath(os.path.join(args.ckpt_dir, 'knowledge.pt'))
    torch.save({'knowledge': knowledge}, kno_path)
    print(f'Save predictions into file {kno_path}.')
    
    evaluator = Evaluator()
    data = dataset[0]
    y_pred = knowledge[-1]

    all_acc = evaluator.eval(y_pred, data.y)['acc']
    train_acc = evaluator.eval(y_pred[data.train_mask], data.y[data.train_mask])['acc']
    val_acc = evaluator.eval(y_pred[data.val_mask], data.y[data.val_mask])['acc']
    test_acc = evaluator.eval(y_pred[data.test_mask], data.y[data.test_mask])['acc']
    print(f'The Prediction, All: {all_acc:.4f}, Train: {train_acc:.4f}, Val: {val_acc:.4f}, Test: {test_acc:.4f}')

# python generate_knowledge.py --ckpt_dir ./ckpt/test_GAT

# args = EasyDict()
# args.ckpt_dir = '/home/changym/competitions/OGB/kd_research/ckpt/test_GAT'