import sys
import time
from collections import namedtuple

import numpy as np
# import scipy as sp
import scipy.sparse as sp

import torch
import torch.nn as nn
from torch.nn import Parameter
import torch.nn.functional as F
import torch.optim as optim

from torch_geometric.datasets import Planetoid
import torch_geometric.transforms as T
from torch_geometric.nn import GCNConv, GATConv
from torch_geometric.nn.inits import glorot, uniform, zeros
from torchsummary import summary

from utils.data_utils import sample_dataset, sample_triplet
from utils.data_utils import batch_generator, batch_generator_one_shot
from utils.metrics import rank_metrics

# torch_geometric/data/data.py
Data = namedtuple('Data', ['x', 'edge_index', 'edge_weight', 'mat'])


class GCN(torch.nn.Module):

    def __init__(self, config):
        super(GCN, self).__init__()

        # model configuration
        self.model_type = config.get('model_type', None)
        print("model type: %s." % self.model_type)

        self.n_node = config['n_node']
        self.feature = config.get('feature', False)
        self.input_dim = config['n_feature'] if self.feature else None
        self.output_dim = 1 if config['n_class'] == 2 else config['n_class']

        # model hyper-parameter
        self.embed_dim = config.get('embed_dim', 32)
        self.conv_dim = config.get('conv_dim', [8, 8])
        self.heads = 8
        self.hidden_dim = config.get('hidden_dim', [64, 32, 16])
        self.dropout = config.get('dropout', 0.0)
        self.mlp = config.get('mlp', True)

        self._build()

    def _build(self):
        """build model.
        https://keras.io/models/model/#compile
        https://keras.io/initializers/
        https://pytorch.org/docs/stable/nn.init
        model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
        """
        if not self.feature:
            self.embedding = nn.Embedding(self.n_node, self.embed_dim)
            nn.init.xavier_uniform_(self.embedding.weight, gain=1.0)
            # nn.init.normal_(self.embedding.weight, std=0.01)

        input_dim = self.input_dim if self.input_dim is not None else self.embed_dim
        self.conv1 = GATConv(input_dim, self.conv_dim[0], heads=self.heads, dropout=0.5)
        # self.conv2 = GATConv(self.conv_dim[0] * 8, self.conv_dim[1], heads=1, concat=True, dropout=0.5)

        if self.mlp:
            self.layer_1 = torch.nn.Linear(self.conv_dim[0] * self.heads * 2, self.hidden_dim[0])
            self.layer_2 = torch.nn.Linear(self.hidden_dim[0], self.hidden_dim[1])
            self.layer_3 = torch.nn.Linear(self.hidden_dim[1], self.hidden_dim[2])
            self.output_layer = torch.nn.Linear(self.hidden_dim[2], self.output_dim)

        return self

    def forward(self, users, items):
        """output = model(input).
        inputs are numpy arrays, and outputs are torch tensors.
        """
        # users, items = inputs
        assert users.size(0) == items.size(0)

        global data
        x, edge_index, edge_weight = data.x, data.edge_index, data.edge_weight

        if not self.feature:
            x = self.embedding(x.type(torch.long))

        h = F.relu(self.conv1(x, edge_index, edge_weight))
        h = F.dropout(h, p=0.5, training=self.training)
        # h = self.conv2(h, edge_index, edge_weight)
        # h = F.dropout(h, p=0.5, training=self.training)
        h_u = torch.index_select(h, 0, users.type(torch.int64).view(-1))
        h_i = torch.index_select(h, 0, items.type(torch.int64).view(-1))
        assert h_u.size(0) == h_i.size(0)

        if not self.mlp:
            # logits = torch.sum(h_u * h_i, dim=1)
            logits = torch.sum(torch.mul(h_u, h_i), dim=1)
        else:
            h = torch.cat([h_u, h_i], dim=1)
            h = torch.relu(self.layer_1(h))
            h = torch.relu(self.layer_2(h))
            h = torch.relu(self.layer_3(h))
            # h = F.dropout(h, p=0.5, training=self.training)
            logits = self.output_layer(h)
            # output = torch.sigmoid(self.output_layer(h))
        return logits


def build_graph_data(mat, config):
    """Building graph data."""
    n_node = config['n_node']
    n_user, n_item = config['n_user'], config['n_item']

    u_zero = sp.csr_matrix((n_user, n_user), dtype=mat.dtype)
    i_zero = sp.csr_matrix((n_item, n_item), dtype=mat.dtype)
    u_mat = sp.hstack([u_zero, mat], format='csr')
    i_mat = sp.hstack([mat.T, i_zero], format='csr')
    ui_mat = sp.vstack([u_mat, i_mat], format='csr')
    assert ui_mat.shape[0] == n_node & ui_mat.shape[1] == n_node

    # only node identity as node feature
    x = np.array(list(range(n_node)))
    # link in the bipartite graph
    row, col = ui_mat.nonzero()
    edge_index = np.vstack((row, col))
    # edge_index = np.vstack((row, col)).transpose()

    # https://github.com/pytorch/pytorch/issues/22615
    x = torch.tensor(x).to(config['device'])
    edge_index = torch.tensor(edge_index.tolist()).to(config['device'])

    return Data(x=x, edge_index=edge_index, edge_weight=None, mat=ui_mat)


def update_graph_data(batch_data, config):
    """Building graph data."""
    global data
    mat = data.mat.copy()
    users, items, label = batch_data
    pos_idx = np.array(label).nonzero()
    mat[np.array(users)[pos_idx], np.array(items)[pos_idx]] = 0
    mat[np.array(items)[pos_idx], np.array(users)[pos_idx]] = 0
    mat.eliminate_zeros()
    row, col = mat.nonzero()
    edge_index = np.vstack((row, col))
    edge_index = torch.tensor(edge_index.tolist()).to(config['device'])
    return Data(x=data.x, edge_index=edge_index, edge_weight=None, mat=data.mat)


def build_train_data(mat, config):
    """Building train data.
    It is necessary to adjust the user/item ids to became graph node ids
    e.g., graph node ids [0, n_user-1] for users, and [n_user, n_user+n_item-1] for items
    """
    users, items, label = sample_dataset(mat, neg_ratio=1)
    items = items + config['n_user']
    # print(type(users), type(items), type(label))
    return users, items, label


def build_train_data_triplet(mat, config):
    """Building triplet train data.
    It is necessary to adjust the user/item ids to became graph node ids
    e.g., graph node ids [0, n_user-1] for users, and [n_user, n_user+n_item-1] for items
    """
    users, pos_items, neg_items = sample_triplet(mat, config['n_user'], config['n_bundle'], batch_size=None)
    pos_items = pos_items + config['n_user']
    neg_items = neg_items + config['n_user']
    # print(type(users), type(pos_items), type(neg_items))
    return users, pos_items, neg_items


def build_test_data(test_data, config):
    """Building test data.
    It is necessary to adjust the user/item ids to became graph node ids
    e.g., graph node ids [0, n_user-1] for users, and [n_user, n_user+n_item-1] for items
    """
    users, items, neg_items = test_data
    items = items + config['n_user']
    neg_items = neg_items + config['n_user']
    return users, items, neg_items


def run(pl, train_mat, test_data, config):

    k = config['top_k']
    n_epochs = config['n_epochs']
    batch_size = config['batch_size']

    # Building test data
    test_data = build_test_data(test_data, config)

    # Evaluating model
    start = time.time()
    hr, mrr, ndcg = rank_metrics(pl, test_data, batch_size, k)
    elapsed = time.time() - start
    log = 'Epoch: {:03d}, Time: {:.4f}, HR@{}: {:.4f}, MRR@{}: {:.4f}, NDCG@{}: {:.4f}'
    print(log.format(0, elapsed, k, hr, k, mrr, k, ndcg))

    best_epoch, best_hr, best_mrr, best_ndcg, = -1, hr, mrr, ndcg

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
        hr, mrr, ndcg = rank_metrics(pl, test_data, batch_size, k)
        elapsed = time.time() - start
        log = 'Epoch: {:03d}, Time: {:.4f}, HR@{}: {:.4f}, MRR@{}: {:.4f}, NDCG@{}: {:.4f}'
        print(log.format(epoch, elapsed, k, hr, k, mrr, k, ndcg))

        if ndcg > best_ndcg:
            best_epoch, best_hr, best_mrr, best_ndcg = epoch, hr, mrr, ndcg

    return best_hr, best_mrr, best_ndcg


def serving(model, input, config):
    pass


if __name__ == '__main__':

    print(sys.version)
    print(sys.version_info)
    print("pytorch version: ", torch.__version__)

    device = torch.device("cpu")

    from bundle.load_data import load_movielens, load_data_steam, load_data_youshu, load_data_netease, load_data_yujian

    # load dataset
    data_dict = load_data_youshu()
    # n_user, n_item = data_dict['n_user'], data_dict['n_item']
    n_user, n_bundle = data_dict['n_user'], data_dict['n_bundle']
    train_mat = data_dict['train_mat']  # user-item interaction matrix
    test_mat = data_dict['test_mat']  # user-item interaction matrix
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
    # config['n_item'] = n_item
    # config['n_node'] = n_user + n_item
    config['n_item'] = n_bundle
    config['n_node'] = n_user + n_bundle
    config['embed_dim'] = 32
    config['triplet'] = False  # using triplet loss
    config['n_epochs'] = 10
    config['batch_size'] = 1024 * 4
    config['device'] = 'cpu'
    config['top_k'] = 5
    config['lr'] = 1e-5  # learning rate
    config['lambda'] = 1e-5  # regularization coefficient

    # global graph data
    data = build_graph_data(train_mat, config)
    print(data)

    # build the model
    model = GCN(config).to(config['device'])

    print('-' * 80)
    print(model)
    print('-' * 80)
    # summary(model, input_size=[(1,), (1,)], device="cpu")

    # define loss function
    # https://pytorch.org/docs/stable/nn.html#crossentropyloss
    # https://pytorch.org/docs/stable/nn.functional.html#binary-cross-entropy-with-logits
    # loss_fn = torch.nn.CrossEntropyLoss().to(config['device'])
    loss_fn = F.binary_cross_entropy_with_logits

    # define an optimizer
    # optimizer = optim.Adam(model.parameters(), lr=config['lr'], weight_decay=config['lambda'])
    optimizer = torch.optim.Adam(model.parameters(), lr=0.05, weight_decay=1e-5)
    # optimizer = torch.optim.Adam(model.parameters(), lr=0.001, betas=(0.9, 0.999), eps=1e-08, weight_decay=0)
    # optimizer = torch.optim.SGD(model.parameters(), lr=4.0)
    # scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 1, gamma=0.9)

    from bundle.pipeline import Pipeline

    # train and evaluate the model
    pl = Pipeline(model, config)
    pl.compile(optimizer, loss_fn, metrics=None)
    hr, mrr, ndcg = run(pl, train_mat, test_data, config)

    print('-' * 80)
    print('HR: {:.4f}, MRR: {:.4f}, NDCG: {:.4f}'.format(hr, mrr, ndcg))
    print('-' * 80)
