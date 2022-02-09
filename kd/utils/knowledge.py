
from typing import List
import torch

from torch import Tensor
import torch.nn.functional as F


def _get_gnn_knowledge(model, x, edge_index, *args, **kwargs):
    """"""
    knos: List[Tensor]= []
    xs: List[Tensor] = []
    for i in range(model.num_layers):
        x = model.convs[i](x, edge_index, *args, **kwargs)
        if (i == model.num_layers - 1 and model.has_out_channels
                and model.jk_mode == 'last'):
            break
        if model.norms is not None:
            x = model.norms[i](x)
        knos.append(x)
        if model.act is not None:
            x = model.act(x)
        x = F.dropout(x, p=model.dropout, training=model.training)
        if hasattr(model, 'jk'):
            xs.append(x)
    x = model.jk(xs) if hasattr(model, 'jk') else x
    x = model.lin(x) if hasattr(model, 'lin') else x
    knos.append(x)
    return knos


def _get_mlp_knowledge(model, x):
    """"""
    knos: List[Tensor] = []
    x = model.lins[0](x)
    knos.append(x)
    for lin, norm in zip(model.lins[1:], model.norms):
        if model.relu_first:
            x = x.relu_()
        x = norm(x)
        if not model.relu_first:
            x = x.relu_()
        x = F.dropout(x, p=model.dropout, training=model.training)
        x = lin.forward(x)
        knos.append(x)
    return knos


def get_knowledge(model, dataset, device, ktype='GNN'):
    model = model.to(device)
    data = dataset[0].to(device)
    with torch.no_grad():
        model.eval()
        if ktype == 'GNN':
            kno = _get_gnn_knowledge(model, data.x, data.edge_index)
        elif ktype == 'MLP':
            kno = _get_mlp_knowledge(model, data.x)
        kno = [x.cpu() for x in kno]
    return kno
