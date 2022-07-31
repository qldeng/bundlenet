
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

# graph data structure
Data = namedtuple('Data', ['x', 'edge_index', 'edge_type', 'n_relation', 'edge_index_full', 'edge_type_full'])


class RGCN(torch.nn.Module):
    """
    RGCN model used to predcit user-bundle preference in tripartite graph.
    """
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
    """
    Building the tripartite graph data.
    """

    n_node = config['n_node']
    n_user, n_bundle, n_item = config['n_user'], config['n_bundle'], config['n_item']
    assert n_node == n_user + n_bundle + n_item

    # only node identity as node feature
    x = np.array(list(range(n_node)))

    mat = data_dict['train_mat']  # user-bundle interaction matrix
    u, b = mat.nonzero()
    b = b + n_user
    edge_index_ub = np.vstack((u, b))
    edge_index_bu = np.vstack((b, u))
    assert edge_index_ub.shape == edge_index_bu.shape

    mat = data_dict['user_item']  # user-item interaction matrix
    u, i = mat.nonzero()
    i = i + n_user + n_bundle
    edge_index_ui = np.vstack((u, i))
    edge_index_iu = np.vstack((i, u))
    assert edge_index_ui.shape == edge_index_iu.shape

    mat = data_dict['bundle_item']  # bundle-item interaction matrix
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
    """
    Update graph in the mini-batch training process.
    """

    global data

    users, bundles, label = batch_data
    bundles = bundles - config['n_user']  # restore the original bundle index
    pos_idx = np.array(label).nonzero()

    # edge_index_ub = np.vstack((users[pos_idx], bundles[pos_idx]))
    # edge_index_bu = np.vstack((bundles[pos_idx], users[pos_idx]))
    # edge_index = np.hstack((edge_index_ub, edge_index_bu))

    mat = data_dict['train_mat'].copy()  # user-bundle interaction matrix
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
