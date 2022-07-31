"""
Reference
https://github.com/rusty1s/pytorch_geometric
https://pytorch-geometric.readthedocs.io/en/latest/
https://pytorch-geometric.readthedocs.io/en/latest/notes/introduction.html#data-handling-of-graphs
"""

import sys
import time
from collections import namedtuple

import numpy as np
import scipy.sparse as sp

import torch
import torch.nn as nn
from torch.nn import Parameter
import torch.nn.functional as F
import torch.optim as optim

from torch_geometric.nn.conv import MessagePassing, RGCNConv
from torch_geometric.nn import GCNConv, GATConv
from torch_geometric.utils import remove_self_loops, add_self_loops, softmax
from torch_geometric.nn.inits import glorot, uniform, zeros

from utils.data_utils import sample_dataset, sample_triplet, batch_generator
from utils.metrics import rank_metrics

Data = namedtuple('Data', ['x', 'edge_index', 'edge_type', 'n_relation', 'edge_index_full', 'edge_type_full'])


class RGATConv(MessagePassing):
    r"""The relational graph attention operator.

    .. math::
        \mathbf{x}^{\prime}_i = \mathbf{\Theta}_{\textrm{root}} \cdot
        \mathbf{x}_i + \sum_{r \in \mathcal{R}} \sum_{j \in \mathcal{N}_r(i)}
        \frac{1}{|\mathcal{N}_r(i)|} \mathbf{\Theta}_r \cdot \mathbf{x}_j,

    where :math:`\mathcal{R}` denotes the set of relations, *i.e.* edge types.
    Edge type needs to be a one-dimensional :obj:`torch.long` tensor which
    stores a relation identifier
    :math:`\in \{ 0, \ldots, |\mathcal{R}| - 1\}` for each edge.

    Args:
        in_channels (int): Size of each input sample.
        out_channels (int): Size of each output sample.
        num_relations (int): Number of relations.
        num_bases (int): Number of bases used for basis-decomposition.
        root_weight (bool, optional): If set to :obj:`False`, the layer will
            not add transformed root node features to the output.
            (default: :obj:`True`)
        bias (bool, optional): If set to :obj:`False`, the layer will not learn
            an additive bias. (default: :obj:`True`)
        **kwargs (optional): Additional arguments of
            :class:`torch_geometric.nn.conv.MessagePassing`.

    Reference
    Relational Graph Attention Networks
    <https://arxiv.org/abs/1904.05811>
    """

    def __init__(self, in_channels, out_channels, num_relations, num_bases,
                 root_weight=True, bias=True, **kwargs):
        super(RGATConv, self).__init__(aggr='add', **kwargs)

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_relations = num_relations
        self.num_bases = num_bases
        self.negative_slope = 0.2
        self.dropout = 0.0

        self.basis = Parameter(torch.Tensor(num_bases, in_channels, out_channels))  # (B, F, F')
        self.weight = Parameter(torch.Tensor(num_relations, num_bases))  # (R, B)
        self.att = Parameter(torch.Tensor(num_relations, 2 * out_channels))  # (R, 2F')

        if root_weight:
            self.root = Parameter(torch.Tensor(in_channels, out_channels))  # (F, F')
        else:
            self.register_parameter('root', None)

        if bias:
            self.bias = Parameter(torch.Tensor(out_channels))  # (F,)
        else:
            self.register_parameter('bias', None)

        self.reset_parameters()

    def reset_parameters(self):
        # size = self.num_bases * self.in_channels
        # uniform(size, self.basis)
        # uniform(size, self.weight)
        # uniform(size, self.att)
        # uniform(size, self.root)
        # uniform(size, self.bias)
        glorot(self.basis)
        glorot(self.weight)
        glorot(self.att)
        glorot(self.root)
        zeros(self.bias)

    def forward(self, x, edge_index, edge_type, edge_norm=None, size=None):
        """"""
        # print("    ", x.shape, edge_index.shape, edge_type.shape, edge_norm, size)

        # print(x.shape, edge_index.shape, edge_type.shape)
        # if torch.is_tensor(x):
        #     edge_index, _ = remove_self_loops(edge_index)
        #     edge_index, _ = add_self_loops(edge_index, num_nodes=x.size(0))
        # print(x.shape, edge_index.shape, edge_type.shape)

        return self.propagate(edge_index, size=size, x=x, edge_type=edge_type, edge_norm=edge_norm)

    def message(self, x_i, x_j, edge_index_i, edge_index_j, edge_type, edge_norm):
        """

        :param x_i: (E, F)
        :param x_j: (E, F)
        :param edge_index_j: (E,)
        :param edge_index_j: (E,)
        :param edge_type: (E,)
        :param edge_norm: (E,)
        :return: (E, F')
        """
        # print("    ", x_i.shape, x_j.shape, edge_index_i.shape, edge_index_j.shape, edge_type.shape, edge_norm)

        w = torch.matmul(self.weight, self.basis.view(self.num_bases, -1))  # (R, F * F')

        if x_i is None:
            w_i = w.view(-1, self.out_channels)
            index = edge_type * self.in_channels + edge_index_i
            h_i = torch.index_select(w_i, 0, index)
        else:
            w_i = w.view(self.num_relations, self.in_channels, self.out_channels)  # (R, F, F')
            w_i = torch.index_select(w_i, 0, edge_type)  # (E, F, F')
            h_i = torch.bmm(x_i.unsqueeze(1), w_i).squeeze(-2)  # (E, F')

        # If no node features are given, we implement a simple embedding
        # loopkup based on the target node index and its edge type.
        if x_j is None:
            w_j = w.view(-1, self.out_channels)
            index = edge_type * self.in_channels + edge_index_j
            h_j = torch.index_select(w_j, 0, index)
        else:
            w_j = w.view(self.num_relations, self.in_channels, self.out_channels)  # (R, F, F')
            w_j = torch.index_select(w_j, 0, edge_type)  # (E, F, F')
            h_j = torch.bmm(x_j.unsqueeze(1), w_j).squeeze(-2)  # (E, F')

        a = self.att.view(self.num_relations, -1)  # (R, 2F')
        a = torch.index_select(a, 0, edge_type)  # (E, 2F')
        alpha = (torch.cat([h_i, h_j], dim=-1) * a).sum(dim=-1)  # (E, 1)

        alpha = F.leaky_relu(alpha, self.negative_slope)

        # Within-Relation Graph Attention
        # adjust edge index i to take relations into account
        edge_index_i = edge_index_i * self.num_relations + edge_type
        # index = edge_index_i.numpy()
        # t = np.unique(index)
        # d = dict(zip(sorted(np.unique(t)), range(len(t))))
        # index = list(map(lambda v: d[v], index))
        # edge_index_i = torch.tensor(index, dtype=edge_index_i.dtype, device=edge_index_i.device)

        # print(edge_index_i.shape, edge_index_i.min(), edge_index_i.max())
        # print(len(edge_index_i.unique(sorted=True)))

        # Across-Relation Graph Attention
        alpha = softmax(alpha, edge_index_i)

        # Sample attention coefficients stochastically.
        alpha = F.dropout(alpha, p=self.dropout, training=self.training)

        h_j = h_j * alpha.view(-1, 1)  # (E, F)

        return h_j if edge_norm is None else h_j * edge_norm.view(-1, 1)  # (E, F')

    def update(self, h, x):
        """

        :param h: (N, F')
        :param x: (N, F)
        :return: (N, F')
        """
        # print("    ", h.shape, x.shape)
        if self.root is not None:
            if x is None:
                h = h + self.root
            else:
                h = h + torch.matmul(x, self.root)
        if self.bias is not None:
            h = h + self.bias
        return h

    def __repr__(self):
        return '{}({}, {}, num_relations={})'.format(
            self.__class__.__name__, self.in_channels, self.out_channels, self.num_relations)


class TGCN(torch.nn.Module):
    def __init__(self, config=None):
        super(TGCN, self).__init__()

        # model configuration
        self.n_node = config['n_node']
        self.n_user = config['n_user']
        self.n_bundle = config['n_bundle']
        self.n_item = config['n_item']
        self.n_relation = config['n_relation']
        self.feature = config.get('feature', False)
        self.input_dim = config['n_feature'] if self.feature else None
        self.output_dim = 1 if config['n_class'] == 2 else config['n_class']

        # model hyper-parameter
        self.embed_dim = config.get('embed_dim', 32)
        self.n_basis = config.get('n_basis', 30)
        self.conv_dim = config.get('conv_dim', [32, 32])
        self.hidden_dim = config.get('hidden_dim', [64, 32, 16])
        self.dropout = config.get('dropout', 0.0)
        self.mlp = config.get('mlp', True)

        if not self.feature:
            self.embedding = torch.nn.Embedding(self.n_node, self.embed_dim)
            nn.init.xavier_uniform_(self.embedding.weight, gain=1.0)
            # nn.init.xavier_normal_(self.embedding.weight, gain=1.0)
            # self.embedding = Parameter(torch.Tensor(self.n_node, self.input_dim))
            # nn.init.xavier_normal_(self.embedding)
            # nn.init.xavier_uniform_(self.embedding)

        input_dim = self.input_dim if self.input_dim is not None else self.embed_dim
        self.conv1 = RGATConv(input_dim, self.conv_dim[0], self.n_relation, self.n_basis)
        self.conv2 = RGATConv(self.conv_dim[0], self.conv_dim[0], self.n_relation, self.n_basis)

        if self.mlp:
            self.layer_1 = torch.nn.Linear(self.conv_dim[0] * 2, self.hidden_dim[0])
            self.layer_2 = torch.nn.Linear(self.hidden_dim[0], self.hidden_dim[1])
            self.layer_3 = torch.nn.Linear(self.hidden_dim[1], self.hidden_dim[2])
            self.output_layer = torch.nn.Linear(self.hidden_dim[2], self.output_dim)

    def forward(self, users, bundles):
        assert len(users) == len(bundles)

        global data
        x, edge_index, edge_type = data.x, data.edge_index, data.edge_type

        if not self.feature:
            x = self.embedding(x.type(torch.long))

        h = F.relu(self.conv1(x, edge_index, edge_type))
        if self.dropout > 0.0:
            h = F.dropout(h, p=self.dropout, training=self.training)
        # h = F.relu(self.conv2(h, edge_index, edge_type))
        # if self.dropout > 0.0:
        #     h = F.dropout(h, p=self.dropout, training=self.training)

        h_u = torch.index_select(h, 0, users.view(-1))
        h_b = torch.index_select(h, 0, bundles.view(-1))
        assert h_u.size(0) == h_b.size(0)

        if not self.mlp:
            # logits = torch.sum(h_u * h_b, dim=1)
            logits = torch.sum(torch.mul(h_u, h_b), dim=1)
        else:
            h = torch.cat([h_u, h_b], dim=1)
            h = torch.relu(self.layer_1(h))
            h = torch.relu(self.layer_2(h))
            h = torch.relu(self.layer_3(h))
            # h = F.dropout(h, p=0.5, training=self.training)
            logits = self.output_layer(h)
        return logits

    def predict(self, inputs, batch_size=None):
        preds = self.forward(inputs)
        return preds.detach().numpy()


def build_graph_data(data_dict, config):
    """Building graph data."""

    n_node = config['n_node']
    n_user, n_bundle, n_item = config['n_user'], config['n_bundle'], config['n_item']
    assert n_node == n_user + n_bundle + n_item

    # only node identity as node feature
    x = np.array(list(range(n_node)))

    mat = data_dict['train_mat']
    u, b = mat.nonzero()
    b = b + n_user
    edge_index_ub = np.vstack((u, b))
    edge_index_bu = np.vstack((b, u))
    assert edge_index_ub.shape == edge_index_bu.shape

    mat = data_dict['user_item']
    u, i = mat.nonzero()
    i = i + n_user + n_bundle
    edge_index_ui = np.vstack((u, i))
    edge_index_iu = np.vstack((i, u))
    assert edge_index_ui.shape == edge_index_iu.shape

    mat = data_dict['bundle_item']
    b, i = mat.nonzero()
    b = b + n_user
    i = i + n_user + n_bundle
    edge_index_bi = np.vstack((b, i))
    edge_index_ib = np.vstack((i, b))
    assert edge_index_bi.shape == edge_index_ib.shape

    print(edge_index_ub.shape, edge_index_ui.shape, edge_index_bi.shape)

    edge_index = [
        edge_index_ub, edge_index_bu,
        edge_index_ui, edge_index_iu,
        edge_index_bi, edge_index_ib
    ]
    n_relation = len(edge_index)
    edge_type = [[i] * (edge_index[i]).shape[1] for i in range(n_relation)]

    edge_index = np.hstack(edge_index)
    edge_type = np.hstack(edge_type)

    print(edge_index.shape, edge_type.shape)
    assert edge_index.shape[0] == edge_index.shape[0]

    x = torch.tensor(x)
    edge_index = torch.tensor(edge_index.tolist())
    edge_type = torch.tensor(edge_type.tolist())
    n_relation = torch.tensor(n_relation)

    return Data(x=x, edge_index=edge_index, edge_type=edge_type, n_relation=n_relation,
                edge_index_full=edge_index, edge_type_full=edge_type)


def update_graph_data(batch_data, config):
    """Building graph data."""
    global data

    users, bundles, label = batch_data
    bundles = bundles - config['n_user']  # restore the original bundle index
    pos_idx = np.array(label).nonzero()

    # edge_index_ub = np.vstack((users[pos_idx], bundles[pos_idx]))
    # edge_index_bu = np.vstack((bundles[pos_idx], users[pos_idx]))
    # edge_index = np.hstack((edge_index_ub, edge_index_bu))

    mat = data_dict['train_bundle'].copy()
    nnz = mat.nnz  # # of edges between users and bundles
    # print(mat.shape, mat.nnz)
    mat[users[pos_idx], bundles[pos_idx]] = 0
    mat.eliminate_zeros()

    assert data.edge_index_full.size(1) == data.edge_type_full.size(0)

    row, col = mat.nonzero()
    edge_index_ub = np.vstack((row, col))
    edge_index_bu = np.vstack((col, row))
    edge_index = np.hstack((edge_index_ub, edge_index_bu))
    edge_index = torch.tensor(edge_index.tolist()).to(config['device'])
    edge_index = torch.cat([edge_index, data.edge_index_full[:,2 * nnz:]], dim=1)

    edge_type = [0] * (edge_index_ub.shape[1]) + [1] * (edge_index_bu.shape[1])
    edge_type = torch.tensor(edge_type).to(config['device'])
    edge_type = torch.cat([edge_type, data.edge_type_full[2 * nnz:]], dim=0)

    # print(edge_index.size(1), edge_type.size(0))
    assert edge_index.size(1) == edge_type.size(0)

    return Data(x=data.x, edge_index=edge_index, edge_type=edge_type, n_relation=data.n_relation,
                edge_index_full=data.edge_index_full, edge_type_full=data.edge_type_full)


def build_train_data(mat, config):
    """Building train data.
    It is necessary to adjust the user/item ids to became graph node ids
    e.g., graph node ids [0, n_user-1] for users, and [n_user, n_user+n_item-1] for items
    """
    users, bundles, label = sample_dataset(mat, neg_ratio=1)
    bundles = bundles + config['n_user']
    return users, bundles, label


def build_train_data_triplet(mat, config):
    """Building triplet train data.
    It is necessary to adjust the user/item ids to became graph node ids
    e.g., graph node ids [0, n_user-1] for users, and [n_user, n_user+n_item-1] for items
    """
    users, pos_bundles, neg_bundles = sample_triplet(mat, config['n_user'], config['n_bundle'], batch_size=None)
    pos_bundles = pos_bundles + config['n_user']
    neg_bundles = neg_bundles + config['n_user']
    return users, pos_bundles, neg_bundles


def build_test_data(test_data, config):
    """Building test data.
    It is necessary to adjust the user/item ids to became graph node ids
    e.g., graph node ids [0, n_user-1] for users, and [n_user, n_user+n_item-1] for items
    """
    users, bundles, neg_bundles = test_data
    bundles = bundles + config['n_user']
    neg_bundles = neg_bundles + config['n_user']
    return users, bundles, neg_bundles


def run(pl, train_mat, test_data, config):

    k = config['top_k']
    n_epochs = config['n_epochs']
    batch_size = config['batch_size']

    # Building test data
    test_data = build_test_data(test_data, config)

    # Evaluating model
    start = time.time()
    model.eval()
    hr, mrr, ndcg = rank_metrics(pl, test_data, batch_size, k)
    elapsed = time.time() - start
    log = 'Epoch: {:03d}, Time: {:.4f}, HR@{}: {:.4f}, MRR@{}: {:.4f}, NDCG@{}: {:.4f}'
    print(log.format(0, elapsed, k, hr, k, mrr, k, ndcg))

    best_epoch, best_hr, best_ndcg, = -1, hr, ndcg

    for epoch in range(1, config['epochs'] + 1):

        # Sampling dataset
        start = time.time()
        # Building train data
        if not config['triplet']:
            train_data = build_train_data(train_mat, config)
        else:
            train_data = build_train_data_triplet(train_mat, config)
        elapsed = time.time() - start
        log = 'Epoch: {:03d}, Time: {:.4f}, Samples: {:.4f}'
        print(log.format(epoch, elapsed, len(train_data[0])))

        # Training model
        start = time.time()
        model.train()
        if not config['triplet']:
            if config['batch_size'] is None:
                hist = pl.fit(train_data, batch_size=None, n_epochs=1)
                loss, accuracy = hist[0]
            else:
                batch_gen = batch_generator(train_data, batch_size=batch_size, shuffle=True, drop_last=True)
                history = []
                steps_per_epoch = int(np.floor(len(train_data[0]) / batch_size))
                # print(len(train_data[0]), config['batch_size'], steps_per_epoch)
                for i in range(steps_per_epoch):
                    batch_data = next(batch_gen)
                    global data
                    data = update_graph_data(batch_data, config)
                    hist = fit(batch_data, batch_size=None, n_epochs=1)
                    loss, accuracy = hist[0]
                    history.append((loss, accuracy))
                history = np.array(history)
                loss, accuracy = np.mean(history, axis=0)
        else:
            hist = pl.fit(train_data, batch_size, n_epochs=1)
            loss, accuracy = hist[0]
        elapsed = time.time() - start
        log = 'Epoch: {:03d}, Time: {:.4f}, Loss: {:.8f}, Accuracy: {:.4f}'
        print(log.format(epoch, elapsed, loss, accuracy))

        # Evaluating model
        start = time.time()
        model.eval()
        hr, mrr, ndcg = rank_metrics(pl, test_data, batch_size, k)
        elapsed = time.time() - start
        log = 'Epoch: {:03d}, Time: {:.4f}, HR@{}: {:.4f}, MRR@{}: {:.4f}, NDCG@{}: {:.4f}'
        print(log.format(epoch, elapsed, k, hr, k, mrr, k, ndcg))

        if ndcg > best_ndcg:
            best_epoch, best_hr, best_mrr, best_ndcg = epoch, hr, mrr, ndcg

    return best_hr, best_mrr, best_ndcg


if __name__ == '__main__':

    # ----------------------------------------------------------------------------------------------------------------

    print(sys.version)
    print(sys.version_info)
    print("pytorch version: ", torch.__version__)

    device = torch.device("cpu")

    from bundle.load_data import load_movielens, load_data_steam, load_data_youshu, load_data_netease, load_data_yujian

    # loading / building dataset
    data_dict = load_data_youshu()
    # n_user, n_item = data_dict['n_user'], data_dict['n_item']
    n_user, n_item, n_bundle = data_dict['n_user'], data_dict['n_item'], data_dict['n_bundle']
    train_mat = data_dict['train_mat']  # train interaction matrix
    test_mat = data_dict['test_mat']  # test interaction matrix
    u_rank, p_rank = test_mat.nonzero()  # Positive user-item interaction
    n_rank = np.array(data_dict['negative'])  # Negative Sampling
    test_data = (u_rank, p_rank, n_rank)  # Triplet test data
    print(type(u_rank), type(p_rank), type(n_rank))
    print(u_rank.shape, p_rank.shape, n_rank.shape)

    # configuration
    config = dict()
    config['model_type'] = None
    config['n_class'] = 2
    config['n_user'] = n_user
    config['n_bundle'] = n_bundle
    config['n_item'] = n_item
    config['n_node'] = n_user + n_bundle + n_item
    config['embed_dim'] = 32
    config['n_basis'] = 16
    config['conv_dim'] = [16, 16]
    config['mlp'] = False
    config['optimizer'] = 'adam'
    config['triplet'] = False  # using triplet loss
    config['n_epochs'] = 10
    config['batch_size'] = None
    config['device'] = 'cpu'
    config['top_k'] = 10

    # global graph data
    data = build_graph_data(data_dict, config)
    print(data)

    config['n_relation'] = data.n_relation

    # build the model
    model = TGCN(config).to(config['device'])

    print('-' * 80)
    print(model)
    print('-' * 80)

    # define loss function
    # https://pytorch.org/docs/stable/nn.html#crossentropyloss
    # https://pytorch.org/docs/stable/nn.functional.html#binary-cross-entropy-with-logits
    # loss_fn = torch.nn.CrossEntropyLoss().to(config['device'])
    loss_fn = F.binary_cross_entropy_with_logits

    # define an optimizer
    # optimizer = torch.optim.Adam(model.parameters(), lr=0.005, weight_decay=1e-5)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=1e-5)

    from bundle.pipeline import Pipeline

    pl = Pipeline(model, config)
    pl.compile(optimizer, loss_fn, metrics=None)

    # train and evaluate the model
    hr, mrr, ndcg = run(pl, train_mat, test_data, config)

    print('-' * 80)
    print('HR: {:.4f}, MRR: {:.4f}, NDCG: {:.4f}'.format(hr, mrr, ndcg))
    print('-' * 80)
