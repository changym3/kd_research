
from collections import defaultdict
from typing import List
import torch

from torch import Tensor
import torch.nn.functional as F


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


def _get_gnn_intermediate_state(model, x, edge_index, *args, **kwargs):
    ks = KnowlegeState()

    xs: List[Tensor] = []
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


def get_model_knowledge(model, data, ktype='GNN', device=None):
    model.eval()
    with torch.no_grad():
        ks = get_model_state(model, data, ktype, device)
    ks = ks.to('cpu')
    return ks


def get_model_state(model, data, stype='GNN', device=None):
    if device is not None:
        model = model.to(device)
        data = data.to(device)
    assert next(model.parameters()).device == data.edge_index.device

    if stype == 'GNN':
        ks = _get_gnn_intermediate_state(model, data.x, data.edge_index)
    elif stype == 'MLP':
        ks = _get_mlp_intermediate_state(model, data.x)
    return ks