import os
from collections import defaultdict

import torch
import torch.nn.functional as F

from kd.data import build_dataset
from kd.trainer import BasicGNNTrainer
from kd.utils.evaluator import Evaluator


class KnowlegeState:
    def __init__(self):
        self._state = defaultdict(list)
    
    def add(self, name, feat):
        self._state[name].append(feat)
    
    def to(self, device):
        state = {}
        for k, feats in self._state.items():
            state[k] = [x.to(device) for x in feats]
        ks = KnowlegeState()
        ks._state = state
        return ks
    
    def __getitem__(self, key):
        return self._state[key]


def get_model_knowledge(model, data, ktype, device=None):
    model.eval()
    with torch.no_grad():
        ks = get_model_state(model, data, ktype, device)
    return ks


def get_model_state(model, data, stype, device=None):
    if device is not None:
        model = model.to(device)
        data = data.to(device)
    assert next(model.parameters()).device == data.edge_index.device

    if stype == 'GAT':
        ks = _get_gnn_intermediate_state(model, data.x, data.edge_index)
    elif stype == 'MLP':
        ks = _get_mlp_intermediate_state(model, data.x)
    return ks


def extract_and_save_knowledge(ckpt_dir, dataset=None):
    '''
        1. recover teacher model from ckpt
        2. extract & save knowledge
        3. evaluate the extracted knowledge 
    '''
    # recover 
    ckpt_path = os.path.join(ckpt_dir, 'model.pt')
    model_state_dict = torch.load(ckpt_path)['model_state_dict']
    cfg_path = os.path.join(ckpt_dir, 'config.pt')  
    cfg = torch.load(cfg_path)['config'] # exp_cfg
    model = BasicGNNTrainer.build_model(cfg)
    model.load_state_dict(model_state_dict)
    # extract
    if dataset is None:
        dataset = build_dataset(cfg.meta.dataset_name)
    device = torch.device(f'cuda:{cfg.trainer.gpu}')
    knowledge = get_model_knowledge(model, dataset[0], device=device).to('cpu')
    # save
    kno_path = os.path.realpath(os.path.join(ckpt_dir, 'knowledge.pt'))
    torch.save({'knowledge': knowledge}, kno_path)
    print(f'Save predictions into file {kno_path}.')
    # evaluate
    evaluator = Evaluator()
    data = dataset[0]
    y_pred = knowledge['feats'][-1].softmax(dim=-1)
    all_acc = evaluator.eval(y_pred, data.y)['acc']
    train_acc = evaluator.eval(y_pred[data.train_mask], data.y[data.train_mask])['acc']
    val_acc = evaluator.eval(y_pred[data.val_mask], data.y[data.val_mask])['acc']
    test_acc = evaluator.eval(y_pred[data.test_mask], data.y[data.test_mask])['acc']
    print(f'The Prediction, All: {all_acc:.4f}, Train: {train_acc:.4f}, Val: {val_acc:.4f}, Test: {test_acc:.4f}')


def _get_gnn_intermediate_state(model, x, edge_index, *args, **kwargs):
    ks = KnowlegeState()

    xs = []
    for i in range(model.num_layers):
        x = model.convs[i](x, edge_index, *args, **kwargs)
        if (i == model.num_layers - 1 and model.has_out_channels
                and model.jk_mode == 'last'):
            break
        if model.norms is not None:
            x = model.norms[i](x)
        ks.add('feats', x)
        if model.act is not None:
            x = model.act(x)
        x = F.dropout(x, p=model.dropout, training=model.training)
        if hasattr(model, 'jk'):
            xs.append(x)
    x = model.jk(xs) if hasattr(model, 'jk') else x
    x = model.lin(x) if hasattr(model, 'lin') else x
    ks.add('feats', x)
    return ks


def _get_mlp_intermediate_state(model, x):
    ks = KnowlegeState()
    x = model.lins[0](x)
    ks.add('feats', x)
    for lin, norm in zip(model.lins[1:], model.norms):
        if model.relu_first:
            x = x.relu_()
        x = norm(x)
        if not model.relu_first:
            x = x.relu_()
        x = F.dropout(x, p=model.dropout, training=model.training)
        x = lin.forward(x)
        ks.add('feats', x)
    return ks
