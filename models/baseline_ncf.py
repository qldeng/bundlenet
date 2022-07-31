"""Neural Collaborative Filtering.
Author: dengqilin@corp.netease.com
Reference
https://github.com/hexiangnan/neural_collaborative_filtering
"""
import sys
import time
import math
import heapq

from collections import namedtuple

import numpy as np
import scipy as sp

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.nn import Parameter

from torchsummary import summary

from utils.data_utils import sample_dataset, sample_triplet
from utils.metrics import rank_metrics


class NCF(torch.nn.Module):

    def __init__(self, config=None):
        super(NCF, self).__init__()

        # model configuration
        self.model_type = config.get('model_type', 'mlp')  # model type (gmf, mlp, ncf)
        print("model type: %s." % self.model_type)

        self.n_user = config['n_user']
        self.n_item = config['n_item']
        self.input_dim = config.get('n_feature', None)
        self.output_dim = 1 if config['n_class'] == 2 else config['n_class']

        # model hyper-parameter
        self.embed_dim = config.get('embed_dim', 32)
        self.hidden_dim = config.get('hidden_dim', [64, 32, 16])
        self.n_factor_gmf = config.get('n_factor_gmf', 8)
        self.n_factor_mlp = config.get('n_factor_mlp', 32)
        self.dropout = config.get('dropout', 0.0)

        self._build()

    def _build(self):
        """build model."""
        if self.model_type == 'gmf':
            self.u_embedding = nn.Embedding(self.n_user, self.n_factor_gmf)
            self.i_embedding = nn.Embedding(self.n_item, self.n_factor_gmf)
            torch.nn.init.xavier_uniform_(self.u_embedding.weight, gain=1.0)
            torch.nn.init.xavier_uniform_(self.i_embedding.weight, gain=1.0)
        elif self.model_type == 'mlp':
            self.u_embedding = nn.Embedding(self.n_user, self.n_factor_mlp)
            self.i_embedding = nn.Embedding(self.n_item, self.n_factor_mlp)
            torch.nn.init.xavier_uniform_(self.u_embedding.weight, gain=1.0)
            torch.nn.init.xavier_uniform_(self.i_embedding.weight, gain=1.0)
        else:
            self.u_embedding_gmf = nn.Embedding(self.n_user, self.n_factor_gmf)
            self.i_embedding_gmf = nn.Embedding(self.n_item, self.n_factor_gmf)
            self.u_embedding_mlp = nn.Embedding(self.n_user, self.n_factor_mlp)
            self.i_embedding_mlp = nn.Embedding(self.n_item, self.n_factor_mlp)
            torch.nn.init.xavier_uniform_(self.u_embedding_gmf.weight, gain=1.0)
            torch.nn.init.xavier_uniform_(self.i_embedding_gmf.weight, gain=1.0)
            torch.nn.init.xavier_uniform_(self.u_embedding_mlp.weight, gain=1.0)
            torch.nn.init.xavier_uniform_(self.i_embedding_mlp.weight, gain=1.0)

        if self.model_type == 'gmf':
            self.output_layer = torch.nn.Linear(self.n_factor_gmf, self.output_dim)
        else:
            self.layer_1 = torch.nn.Linear(self.n_factor_mlp * 2, self.hidden_dim[0])
            self.layer_2 = torch.nn.Linear(self.hidden_dim[0], self.hidden_dim[1])
            self.layer_3 = torch.nn.Linear(self.hidden_dim[1], self.hidden_dim[2])
            self.output_layer = torch.nn.Linear(self.hidden_dim[2], self.output_dim)
        return self

    def forward(self, x_u, x_i):
        """output = model(input).
        inputs are numpy arrays, and outputs are torch tensors.
        """
        # x_u, x_i = inputs
        assert x_u.size(0) == x_i.size(0)
        if self.model_type == 'gmf':
            h_u = self.u_embedding(x_u).squeeze(dim=1)  # (batch_size, embed_dim)
            h_i = self.i_embedding(x_i).squeeze(dim=1)  # (batch_size, embed_dim)
            assert h_u.size(0) == h_i.size(0)
            h = torch.mul(h_u, h_i)
            logits = self.output_layer(h)
        elif self.model_type == 'mlp':
            h_u = self.u_embedding(x_u).squeeze(dim=1)  # (batch_size, embed_dim)
            h_i = self.i_embedding(x_i).squeeze(dim=1)  # (batch_size, embed_dim)
            assert h_u.size(0) == h_i.size(0)
            h = torch.cat([h_u, h_i], dim=-1)
            h = torch.relu(self.layer_1(h))
            h = torch.relu(self.layer_2(h))
            h = torch.relu(self.layer_3(h))
            # h = F.dropout(h, p=0.5, training=self.training)
            # output = torch.sigmoid(self.output_layer(h))
            logits = self.output_layer(h)
        else:
            h_u_gmf = self.u_embedding_gmf(x_u).squeeze(dim=1)  # (batch_size, embed_dim)
            h_i_gmf = self.i_embedding_gmf(x_i).squeeze(dim=1)  # (batch_size, embed_dim)
            h_u_mlp = self.u_embedding_mlp(x_u).squeeze(dim=1)  # (batch_size, embed_dim)
            h_i_mlp = self.i_embedding_mlp(x_i).squeeze(dim=1)  # (batch_size, embed_dim)
            h_gmf = torch.mul(h_u_gmf, h_i_gmf)
            h = torch.cat([h_u_mlp, h_i_mlp], dim=-1)
            h = torch.relu(self.layer_1(h))
            h = torch.relu(self.layer_2(h))
            h_mlp = torch.relu(self.layer_3(h))
            h = torch.cat([h_gmf, h_mlp], dim=-1)
            # h = F.dropout(h, p=0.5, training=self.training)
            # output = torch.sigmoid(self.output_layer(h))
            logits = self.output_layer(h)
        return logits


def run(pl, train_mat, test_data, config):

    k = config['top_k']
    n_epochs = config['n_epochs']
    batch_size = config['batch_size']

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
        if not config['triplet']:
            train_data = sampling_dataset(train_mat, neg_ratio=5)
        else:
            train_data = sampling_triplet(train_mat, config['n_user'], config['n_bundle'])
        elapsed = time.time() - start
        log = 'Epoch: {:03d}, Time: {:.4f}, Samples: {:.4f}'
        print(log.format(epoch, elapsed, len(train_data[0])))

        # Training model
        start = time.time()
        if not config['triplet']:
            hist = pl.fit(train_data, batch_size, n_epochs=1)
            loss, accuracy = hist[0]
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


def serving(model, inputs, config):
    pass


if __name__ == '__main__':

    # ----------------------------------------------------------------------------------------------------------------

    print(sys.version)
    print(sys.version_info)
    print("pytorch version: ", torch.__version__)

    device = torch.device("cpu")

    from load_data import load_data_ml_1m, load_data_steam, load_data_youshu, load_data_netease, load_data_yujian

    # load dataset
    data_dict = load_data_ml_1m()
    n_user, n_item = data_dict['n_user'], data_dict['n_item']
    train_mat = data_dict['train_mat']  # train interaction matrix
    test_mat = data_dict['test_mat']  # test interaction matrix
    u_rank, p_rank = test_mat.nonzero()  # Positive user-item interaction
    n_rank = np.array(data_dict['negative'])  # Negative Sampling
    test_data = (u_rank, p_rank, n_rank)  # Triplet test data
    print(type(u_rank), type(p_rank), type(n_rank))
    print(u_rank.shape, p_rank.shape, n_rank.shape)

    # configuration
    config = dict()
    config['model_type'] = 'mlp'  # model type (gmf, mlp, ncf)
    config['n_class'] = 2
    config['n_user'] = n_user
    config['n_item'] = n_item
    config['embed_dim'] = 32
    config['triplet'] = False  # using triplet loss
    config['n_epochs'] = 5
    config['batch_size'] = 1024
    config['device'] = 'cpu'
    config['top_k'] = 5
    config['lr'] = 1e-5  # learning rate
    config['lambda'] = 1e-5  # regularization coefficient

    # build the model
    model = NCF(config).to(config['device'])

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
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001, betas=(0.9, 0.999), eps=1e-08, weight_decay=0)
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
