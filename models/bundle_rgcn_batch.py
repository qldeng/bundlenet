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

from utils.data_utils import sample_dataset, sample_triplet, batch_generator, batch_generator_one_shot
from utils.metrics import rank_metrics

Data = namedtuple('Data', ['x', 'edge_index', 'edge_type', 'n_relation', 'edge_index_full', 'edge_type_full'])


class RGCN(torch.nn.Module):
    def __init__(self, config=None):
        super(RGCN, self).__init__()

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
        self.n_basis = config.get('n_basis', 16)
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
        self.conv1 = RGCNConv(input_dim, self.conv_dim[0], self.n_relation, self.n_basis)
        self.conv2 = RGCNConv(self.conv_dim[0], self.conv_dim[0], self.n_relation, self.n_basis)
        # self.conv3 = RGCNConv(self.conv_dim[0], self.conv_dim[0], self.n_relation, self.n_basis)

        if self.mlp:
            self.fc1 = torch.nn.Linear(self.conv_dim[0] * 2, self.hidden_dim[0])
            self.fc2 = torch.nn.Linear(self.hidden_dim[0], self.hidden_dim[1])
            self.fc3 = torch.nn.Linear(self.hidden_dim[1], self.hidden_dim[2])
            self.output = torch.nn.Linear(self.hidden_dim[2], self.output_dim)

    def forward(self, users, bundles):
        assert len(users) == len(bundles)

        global data
        x, edge_index, edge_type = data.x, data.edge_index, data.edge_type

        if not self.feature:
            x = self.embedding(x.type(torch.long))

        h = F.leaky_relu(self.conv1(x, edge_index, edge_type))
        if self.dropout > 0.0:
            h = F.dropout(h, p=self.dropout, training=self.training)
        h = F.leaky_relu(self.conv2(h, edge_index, edge_type))
        if self.dropout > 0.0:
            h = F.dropout(h, p=self.dropout, training=self.training)
        # h = F.relu(self.conv3(h, edge_index, edge_type))
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
            h = F.leaky_relu(self.fc1(h))
            h = F.leaky_relu(self.fc2(h))
            h = F.leaky_relu(self.fc3(h))
            # h = F.dropout(h, p=0.5, training=self.training)
            logits = self.output(h)
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
    assert edge_index.shape[1] == edge_type.shape[0]

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

    mat = data_dict['train_mat'].copy()
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
    hr, mrr, ndcg = rank_metrics(pl, test_data, batch_size=None, k=k)
    elapsed = time.time() - start
    log = 'Epoch: {:03d}, Time: {:.4f}, HR@{}: {:.4f}, MRR@{}: {:.4f}, NDCG@{}: {:.4f}'
    print(log.format(0, elapsed, k, hr, k, mrr, k, ndcg))

    best_epoch, best_hr, best_ndcg, = -1, hr, ndcg

    for epoch in range(1, n_epochs + 1):

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
                for i in range(steps_per_epoch):
                    batch_data = next(batch_gen)
                    global data
                    data = update_graph_data(batch_data, config)
                    hist = pl.fit(batch_data, batch_size=None, n_epochs=1)
                    loss, accuracy = hist[0]
                    history.append((loss, accuracy))
                    log = '\tStep: {:d}/{:d}, Loss: {:.8f}, Accuracy: {:.4f}'
                    print(log.format(i + 1, steps_per_epoch, loss, accuracy))
                history = np.array(history)
                loss, accuracy = np.mean(history, axis=0)
        else:
            hist = pl.fit_triplet(train_data, batch_size, n_epochs=1)
            loss, accuracy = hist[0]
        elapsed = time.time() - start
        log = 'Epoch: {:03d}, Time: {:.4f}, Loss: {:.8f}, Accuracy: {:.4f}'
        print(log.format(epoch, elapsed, loss, accuracy))

        # Evaluating model
        start = time.time()
        model.eval()
        hr, mrr, ndcg = rank_metrics(pl, test_data, batch_size=None, k=k)
        elapsed = time.time() - start
        log = 'Epoch: {:03d}, Time: {:.4f}, HR@{}: {:.4f}, MRR@{}: {:.4f}, NDCG@{}: {:.4f}'
        print(log.format(epoch, elapsed, k, hr, k, mrr, k, ndcg))

        if ndcg > best_ndcg:
            best_epoch, best_hr, best_mrr, best_ndcg = epoch, hr, mrr, ndcg

    return best_hr, best_mrr, best_ndcg


if __name__ == '__main__':

    print(sys.version)
    print(sys.version_info)
    print("pytorch version: ", torch.__version__)

    device = torch.device("cpu")

    from bundle.load_data import load_movielens, load_data_steam, load_data_youshu, load_data_netease, load_data_yujian

    # load dataset
    data_dict = load_data_netease()
    # n_user, n_item = data_dict['n_user'], data_dict['n_item']
    n_user, n_item, n_bundle = data_dict['n_user'], data_dict['n_item'], data_dict['n_bundle']
    train_mat = data_dict['train_mat']  # train interaction matrix
    test_mat = data_dict['test_mat']  # test interaction matrix
    u_rank, p_rank = test_mat.nonzero()  # Positive user-item interaction
    n_rank = np.array(data_dict['negative'])  # Negative Sampling
    test_data = (u_rank, p_rank, n_rank)  # Triplet test data
    # print(type(u_rank), type(p_rank), type(n_rank))
    # print(u_rank.shape, p_rank.shape, n_rank.shape)
    print(n_user, n_item, n_bundle)
    print(train_mat.nnz, test_mat.nnz)

    # configuration
    config = dict()
    config['model_type'] = None
    config['n_class'] = 2
    config['n_user'] = n_user
    config['n_bundle'] = n_bundle
    config['n_item'] = n_item
    config['n_node'] = n_user + n_bundle + n_item
    config['embed_dim'] = 16
    config['n_basis'] = 16
    config['conv_dim'] = [16, 16]
    config['mlp'] = True
    config['optimizer'] = 'adam'
    config['triplet'] = False  # using triplet loss
    config['n_epochs'] = 20
    config['batch_size'] = 1024 * 10
    config['device'] = 'cpu'
    config['top_k'] = 5

    # global graph data
    data = build_graph_data(data_dict, config)
    print(data)

    config['n_relation'] = data.n_relation

    for k in config:
        print("{}: {}".format(k, config[k]))

    # build the model
    model = RGCN(config).to(config['device'])

    print('-' * 80)
    print(model)
    print('-' * 80)

    # define loss function
    # https://pytorch.org/docs/stable/nn.html#crossentropyloss
    # https://pytorch.org/docs/stable/nn.functional.html#binary-cross-entropy-with-logits
    # loss_fn = torch.nn.CrossEntropyLoss().to(config['device'])
    loss_fn = F.binary_cross_entropy_with_logits

    # define an optimizer
    # optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=1e-5)

    from bundle.pipeline import Pipeline

    # train and evaluate the model
    pl = Pipeline(model, config)
    pl.compile(optimizer, loss_fn, metrics=None)
    hr, mrr, ndcg = run(pl, train_mat, test_data, config)

    print('-' * 80)
    print('HR: {:.4f}, MRR: {:.4f}, NDCG: {:.4f}'.format(hr, mrr, ndcg))
    print('-' * 80)
