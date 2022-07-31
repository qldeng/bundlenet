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

from torch_geometric.nn.conv import MessagePassing
from torch_geometric.nn import GCNConv, GATConv

from utils.data_utils import sample_dataset, sample_triplet, batch_generator, batch_generator_one_shot
from utils.metrics import rank_metrics

Data = namedtuple('Data', ['x', 'edge_index_ub', 'edge_index_ui', 'edge_index_bi', 'ub_mat'])


class GCNTri(torch.nn.Module):
    def __init__(self, config=None):
        super(GCNTri, self).__init__()

        # model configuration
        self.n_node = config['n_node']
        self.n_user = config['n_user']
        self.n_bundle = config['n_bundle']
        self.n_item = config['n_item']
        self.feature = config.get('feature', False)
        self.input_dim = config['n_feature'] if self.feature else None
        self.output_dim = 1 if config['n_class'] == 2 else config['n_class']

        # model hyper-parameter
        self.embed_dim = config.get('embed_dim', 32)
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
        self.ub_conv1 = GCNConv(input_dim, self.conv_dim[0], cached=False)
        self.ui_conv1 = GCNConv(input_dim, self.conv_dim[0], cached=False)
        self.bi_conv1 = GCNConv(input_dim, self.conv_dim[0], cached=False)
        self.ub_conv2 = GCNConv(self.conv_dim[0], self.conv_dim[1], cached=False)
        self.ui_conv2 = GCNConv(self.conv_dim[0], self.conv_dim[1], cached=False)
        self.bi_conv2 = GCNConv(self.conv_dim[0], self.conv_dim[1], cached=False)

        if self.mlp:
            self.fc1 = torch.nn.Linear(self.conv_dim[0] * 2, self.hidden_dim[0])
            self.fc2 = torch.nn.Linear(self.hidden_dim[0], self.hidden_dim[1])
            self.fc3 = torch.nn.Linear(self.hidden_dim[1], self.hidden_dim[2])
            self.output = torch.nn.Linear(self.hidden_dim[2], self.output_dim)

    def forward(self, users, bundles):
        """
        :param users: (batch_size, 1)
        :param bundles: (batch_size, 1)
        :return: (batch_size, 1)
        """
        assert users.size(0) == bundles.size(0)
        # assert len(users) == len(bundles)
        # print(users.shape, bundles.shape)

        global data
        x = data.x

        x_ub = torch.cat((x[:self.n_user], x[self.n_user:self.n_user + self.n_bundle]), 0)
        x_ui = torch.cat((x[:self.n_user], x[self.n_user + self.n_bundle:]), 0)
        x_bi = torch.cat((x[self.n_user:self.n_user + self.n_bundle], x[self.n_user + self.n_bundle:]), 0)

        if not self.feature:
            x_ub = self.embedding(x_ub.type(torch.long))
            x_ui = self.embedding(x_ui.type(torch.long))
            x_bi = self.embedding(x_bi.type(torch.long))
            # x_ub = torch.index_select(self.embedding, 0, x_ub.type(torch.int64))
            # x_ui = torch.index_select(self.embedding, 0, x_ui.type(torch.int64))
            # x_bi = torch.index_select(self.embedding, 0, x_bi.type(torch.int64))
        # print(x_ub.shape, x_ui.shape, x_bi.shape)

        h_ub = F.elu(self.ub_conv1(x_ub, data.edge_index_ub))
        if self.dropout > 0.0:
            h_ub = F.dropout(h_ub, p=self.dropout, training=self.training)
        h_ui = F.elu(self.ui_conv1(x_ui, data.edge_index_ui))
        if self.dropout > 0.0:
            h_ui = F.dropout(h_ui, p=self.dropout, training=self.training)
        h_bi = F.elu(self.bi_conv1(x_bi, data.edge_index_bi))
        if self.dropout > 0.0:
            h_bi = F.dropout(h_bi, p=self.dropout, training=self.training)

        h_ub = torch.cat((h_ub[:self.n_user], h_ub[self.n_user:], torch.zeros(self.n_item, h_ub.size(1))), 0)
        h_ui = torch.cat((h_ui[:self.n_user], torch.zeros(self.n_bundle, h_ui.size(1)), h_ui[self.n_user:]), 0)
        h_bi = torch.cat((torch.zeros(self.n_user, h_bi.size(1)), h_bi[:self.n_bundle], h_bi[self.n_bundle:]), 0)

        h = (h_ub + h_ui + h_bi) / 2

        h_ub = torch.cat((h[:self.n_user], h[self.n_user:self.n_user + self.n_bundle]), 0)
        h_ui = torch.cat((h[:self.n_user], h[self.n_user + self.n_bundle:]), 0)
        h_bi = torch.cat((h[self.n_user:self.n_user + self.n_bundle], h[self.n_user + self.n_bundle:]), 0)

        h_ub = F.elu(self.ub_conv2(h_ub, data.edge_index_ub))
        if self.dropout > 0.0:
            h_ub = F.dropout(h_ub, p=self.dropout, training=self.training)
        h_ui = F.elu(self.ui_conv2(h_ui, data.edge_index_ui))
        if self.dropout > 0.0:
            h_ui = F.dropout(h_ui, p=self.dropout, training=self.training)
        h_bi = F.elu(self.bi_conv2(h_bi, data.edge_index_bi))
        if self.dropout > 0.0:
            h_bi = F.dropout(h_bi, p=self.dropout, training=self.training)

        h_ub = torch.cat((h_ub[:self.n_user], h_ub[self.n_user:], torch.zeros(self.n_item, h_ub.size(1))), 0)
        h_ui = torch.cat((h_ui[:self.n_user], torch.zeros(self.n_bundle, h_ui.size(1)), h_ui[self.n_user:]), 0)
        h_bi = torch.cat((torch.zeros(self.n_user, h_bi.size(1)), h_bi[:self.n_bundle], h_bi[self.n_bundle:]), 0)

        h = (h_ub + h_ui + h_bi) / 2

        h_u = torch.index_select(h, 0, users.view(-1))
        h_b = torch.index_select(h, 0, bundles.view(-1))
        assert h_u.size(0) == h_b.size(0)

        if not self.mlp:
            # logits = torch.sum(h_u * h_b, dim=1)
            logits = torch.sum(torch.mul(h_u, h_b), dim=1)
        else:
            h = torch.cat([h_u, h_b], dim=1)
            h = torch.relu(self.fc1(h))
            h = torch.relu(self.fc2(h))
            h = torch.relu(self.fc3(h))
            # h = F.dropout(h, p=0.5, training=self.training)
            logits = self.output(h)
        return logits

    def predict(self, inputs, batch_size=None):
        preds = self.forward(inputs)
        return preds.detach().numpy()


def build_graph_data(data_dict, config):
    """Building graph data."""

    n_node = config['n_node']
    n_user, n_bundle, n_item, = config['n_user'], config['n_bundle'], config['n_item']
    n_node_ub = n_user + n_bundle
    n_node_ui = n_user + n_item
    n_node_bi = n_bundle + n_item

    u_zero = sp.csr_matrix((n_user, n_user))
    b_zero = sp.csr_matrix((n_bundle, n_bundle))
    i_zero = sp.csr_matrix((n_item, n_item))

    # only node identity as node feature
    x = np.array(list(range(n_node)))

    mat = data_dict['train_mat']
    u_mat = sp.hstack([u_zero, mat], format='csr')
    b_mat = sp.hstack([mat.T, b_zero], format='csr')
    ub_mat = sp.vstack([u_mat, b_mat], format='csr')
    assert ub_mat.shape[0] == n_node_ub & ub_mat.shape[1] == n_node_ub
    # link in the user-bundle bipartite graph
    edge_index_ub = np.vstack(ub_mat.nonzero())

    mat = data_dict['user_item']
    u_mat = sp.hstack([u_zero, mat], format='csr')
    i_mat = sp.hstack([mat.T, i_zero], format='csr')
    ui_mat = sp.vstack([u_mat, i_mat], format='csr')
    assert ui_mat.shape[0] == n_node_ui & ui_mat.shape[1] == n_node_ui
    # link in the user-item bipartite graph
    edge_index_ui = np.vstack(ui_mat.nonzero())

    mat = data_dict['bundle_item']
    b_mat = sp.hstack([b_zero, mat], format='csr')
    i_mat = sp.hstack([mat.T, i_zero], format='csr')
    bi_mat = sp.vstack([b_mat, i_mat], format='csr')
    assert bi_mat.shape[0] == n_node_bi & bi_mat.shape[1] == n_node_bi
    # link in the bundle-item bipartite graph
    edge_index_bi = np.vstack(bi_mat.nonzero())

    x = torch.tensor(x)
    edge_index_ub = torch.tensor(edge_index_ub.tolist())
    edge_index_ui = torch.tensor(edge_index_ui.tolist())
    edge_index_bi = torch.tensor(edge_index_bi.tolist())

    return Data(x=x, edge_index_ub=edge_index_ub, edge_index_ui=edge_index_ui, edge_index_bi=edge_index_bi,
                ub_mat=ub_mat)


def update_graph_data(batch_data, config):
    """Building graph data."""
    global data
    mat = data.ub_mat.copy()
    users, bundles, label = batch_data
    pos_idx = np.array(label).nonzero()
    mat[np.array(users)[pos_idx], np.array(bundles)[pos_idx]] = 0
    mat[np.array(bundles)[pos_idx], np.array(users)[pos_idx]] = 0
    mat.eliminate_zeros()
    row, col = mat.nonzero()
    edge_index = np.vstack((row, col))
    edge_index = torch.tensor(edge_index.tolist()).to(config['device'])
    return Data(x=data.x, edge_index_ub=edge_index, edge_index_ui=data.edge_index_ui, edge_index_bi=data.edge_index_bi,
                ub_mat=data.ub_mat)


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

    for k in config:
        print("{}: {}".format(k, config[k]))

    # build the model
    model = GCNTri(config).to(config['device'])

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
