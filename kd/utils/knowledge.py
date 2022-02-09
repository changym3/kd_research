
from typing import List
import torch

from torch import Tensor
import torch.nn.functional as F


def _get_gnn_knowledge(self, x, edge_index, *args, **kwargs):
    """"""
    xs: List[Tensor] = []
    for i in range(self.num_layers):
        x = self.convs[i](x, edge_index, *args, **kwargs)
        if (i == self.num_layers - 1 and self.has_out_channels
                and self.jk_mode == 'last'):
            break
        if self.norms is not None:
            x = self.norms[i](x)
        if self.act is not None:
            x = self.act(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        xs.append(x)
    x = self.jk(xs) if hasattr(self, 'jk') else x
    x = self.lin(x) if hasattr(self, 'lin') else x
    xs.append(x)
    xs = [x.cpu() for x in xs]
    return xs


def get_knowledge(model, dataset, device):
    model = model.to(device)
    data = dataset[0].to(device)
    with torch.no_grad():
        model.eval()
        kno = _get_gnn_knowledge(model, data.x, data.edge_index)
    return kno