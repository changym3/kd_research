import copy
import torch
import torch_geometric
from torch_geometric.transforms import BaseTransform
from torch_sparse import SparseTensor




class AugmentedFeatures:
    def __init__(self, cfg):
        self.cfg = cfg
        self.knn_k = cfg.model.knn.k
        self.knn_hop = cfg.model.knn.hop
        self.knn_merge_graph = cfg.model.knn.merge_graph
        self.cosine = cfg.model.knn.cosine 
        self.raw_hop = cfg.model.raw.hop

    def augment_features(self, data, knowledge):
        if self.knn_hop > 0:
            if self.cfg.model.knn.pos == 'logit':
                pos = knowledge['feats'][-1]
            elif self.cfg.model.knn.pos == 'hidden':
                pos = knowledge['feats'][0]
            knn_graph = self.augment_knn_graph(data, pos, self.knn_k, self.cosine)
            knn_x_list = self.sign_feats(knn_graph, self.knn_hop, norm=True)[1:]
        else:
            knn_x_list = []

        if self.raw_hop > 0:
            raw_x_list = self.sign_feats(data, self.raw_hop, norm=True)
        else:
            raw_x_list = [data.x]

        feats = torch.stack(raw_x_list + knn_x_list , dim=0)

        # if self.soft_enable:
        #     soft_data = self.augment_soft_graph(data, knowledge)
        #     soft_x_list = self.sign_feats(soft_data, self.raw_hop, norm=True)
        #     soft_feats = torch.stack(soft_x_list, dim=0)
        #     feats = torch.cat([feats, soft_feats], dim=-1)
        #     self.cfg.dataset.num_features += self.cfg.dataset.num_classes
        return feats
       
    def augment_knn_graph(self, data, pos, k, cosine):
        device = data.x.device
        pos = pos.to(device)
        knn_data = copy.deepcopy(data)
        knn_edges = torch_geometric.nn.knn_graph(x=pos, k=k, cosine=cosine, loop=True).to(device)
        if self.knn_merge_graph:
            knn_data.edge_index = torch.cat([knn_data.edge_index, knn_edges], dim=-1)
        else:
            knn_data.edge_index = knn_edges
        return knn_data

    def sign_feats(self, data, aug_k, norm):
        data = SIGN(aug_k, norm)(data)
        x_list = [data.x]
        for i in range(1, aug_k+1):
            x_list.append(getattr(data, f'x{i}'))
            delattr(data, f'x{i}')
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