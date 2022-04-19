import copy
import torch
import torch_geometric
from torch_geometric.transforms import BaseTransform
from torch_sparse import SparseTensor


def sign_feats(data, aug_k, norm=True):
    data = SIGN(aug_k, norm)(data)
    x_list = [data.x]
    for i in range(1, aug_k+1):
        x_list.append(getattr(data, f'x{i}'))
        delattr(data, f'x{i}')
    return x_list


def aug_feat_to_path(data, aug_max, save_path):
    data_tmp = data.to('cpu')
    x_list = sign_feats(data_tmp, aug_k=aug_max)
    torch.save({'x_list': x_list}, save_path)

class AugmentedFeatures:
    def __init__(self, aug_hop, aug_path=None):
        self.aug_hop = aug_hop
        self.aug_path = aug_path

    def augment_features(self, data):
        device = data.x.device
        if self.aug_path:
            x_list = torch.load(self.aug_path)['x_list']
            x_list = x_list[:self.aug_hop+1]
            x_list = [x.to(device) for x in x_list]
        else:
            if self.aug_hop > 0:
                x_list = sign_feats(data, self.aug_hop, norm=True)
            else:
                x_list = [data.x]
        return x_list
    

    



class SIGN(BaseTransform):
    r"""The Scalable Inception Graph Neural Network module (SIGN) from the
    `"SIGN: Scalable Inception Graph Neural Networks"
    <https://arxiv.org/abs/2004.11198>`_ paper, which precomputes the fixed
    representations

    .. math::
        \mathbf{X}^{(i)} = {\left( \mathbf{D}^{-1/2} \mathbf{A}
        \mathbf{D}^{-1/2} \right)}^i \mathbf{X}

    for :math:`i \in \{ 1, \ldots, K \}` and saves them in
    :obj:`data.x1`, :obj:`data.x2`, ...

    .. note::

        Since intermediate node representations are pre-computed, this operator
        is able to scale well to large graphs via classic mini-batching.
        For an example of using SIGN, see `examples/sign.py
        <https://github.com/pyg-team/pytorch_geometric/blob/master/examples/
        sign.py>`_.

    Args:
        K (int): The number of hops/layer.
    """
    def __init__(self, K, norm):
        self.K = K
        self.norm = norm

    def __call__(self, data):
        assert data.edge_index is not None
        row, col = data.edge_index
        adj_t = SparseTensor(row=col, col=row,
                             sparse_sizes=(data.num_nodes, data.num_nodes))
        if self.norm:
            deg = adj_t.sum(dim=1).to(torch.float)
            deg_inv_sqrt = deg.pow(-0.5)
            deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0
            adj_t = deg_inv_sqrt.view(-1, 1) * adj_t * deg_inv_sqrt.view(1, -1)

        assert data.x is not None
        xs = [data.x]
        for i in range(1, self.K + 1):
            xs += [adj_t @ xs[-1]]
            data[f'x{i}'] = xs[-1]

        return data

    def __repr__(self) -> str:
        return f'{self.__class__.__name__}(K={self.K})'