import time
import pickle

import numpy as np
from scipy import sparse, io, stats
import pandas as pd
import scipy.sparse as sp

import torch
from torch import nn
import torch.nn.functional as F
from torch import optim

from torch.nn import Parameter
from torch.autograd import Variable

from tqdm import tqdm

from utils.data_utils import sample_batch, batch_generator, batch_generator_one_shot


class DAM(torch.nn.Module):

    def __init__(self, config):
        super(DAM, self).__init__()

        # model configuration
        self.n_user = config['n_user']
        self.n_item = config['n_item']
        self.n_bundle = config['n_bundle']

        # model hyper-parameter
        self.embed_dim = config.get('embed_dim', 16)
        self.hidden_dim = config.get('hidden_dim', [64, 32, 16])
        self.dropout = config.get('dropout', 0.5)

        self.embedding_u = nn.Embedding(self.n_user, self.embed_dim)
        self.embedding_i = nn.Embedding(self.n_item, self.embed_dim)
        self.embedding_b = nn.Embedding(self.n_bundle, self.embed_dim)
        self.A = Parameter(torch.Tensor(self.n_item, self.embed_dim))
        # nn.init.xavier_normal_(self.embedding_u.weight)
        # nn.init.xavier_normal_(self.embedding_i.weight)
        # nn.init.xavier_normal_(self.embedding_b.weight)
        # nn.init.xavier_normal_(self.A)
        nn.init.kaiming_uniform_(self.embedding_u.weight)
        nn.init.kaiming_uniform_(self.embedding_i.weight)
        nn.init.kaiming_uniform_(self.embedding_b.weight)
        nn.init.xavier_uniform_(self.A)

        self.fc1 = nn.Linear(self.embed_dim * 2, self.embed_dim * 2)
        self.fc2 = nn.Linear(self.embed_dim * 2, self.embed_dim * 2)
        self.output_ub = nn.Linear(self.embed_dim * 2, 1)
        self.output_ui = nn.Linear(self.embed_dim * 2, 1)

        print('-' * 80)
        # params = list(self.parameters())
        for param in self.parameters():
            print(type(param.data), param.size())
        print('-' * 80)

    def forward(self, x_u, x_i):
        h_u = self.embedding_u(x_u).squeeze(dim=1)  # (batch_size, embed_dim)
        h_i = self.embedding_i(x_i).squeeze(dim=1)  # (batch_size, embed_dim)
        h = torch.cat((h_u, h_i), dim=1)
        h = F.leaky_relu(self.fc1(h))
        # h = F.dropout(h, p=0.0, training=self.training)
        h = F.leaky_relu(self.fc2(h))
        # h = F.dropout(h, p=0.0, training=self.training)
        h = self.output_ui(h)
        return h

    def forward_bundle(self, x_u, x_b):
        h_u = self.embedding_u(x_u).squeeze(dim=1)  # (batch_size, embed_dim)
        h_b = self.embedding_b(x_b).squeeze(dim=1)  # (batch_size, embed_dim)
        h_x = self.attention(x_u, x_b)
        h = torch.cat((h_u, h_b + h_x), dim=1)
        h = F.leaky_relu(self.fc1(h))
        # h = F.dropout(h, p=0.5, training=self.training)
        h = F.leaky_relu(self.fc2(h))
        # h = F.dropout(h, p=0.5, training=self.training)
        h = self.output_ub(h)
        return h

    def attention(self, x_u, x_b):
        """Factorized Attention Network."""
        h = []
        for i, b in enumerate(x_b):
            items = bundle_item[b].nonzero()[1]
            items = torch.tensor(items, dtype=torch.long)
            embeds = self.embedding_i(items)
            # print(type(embeds), embeds.shape)
            w = F.softmax(torch.sum(self.embedding_u(x_u[i].expand(len(items))) * self.A[items], 1, True), 0)
            # h.append(torch.sum(embeds * w, 0) + self.embedding_b(b))
            h.append(torch.sum(embeds * w, 0))
        h = torch.stack(h, dim=0)
        return h

    def predict(self, x_u, x_i):
        preds = self.forward(x_u, x_i)
        return preds.detach().numpy()

    def predict_bundle(self, x_u, x_b):
        preds = self.forward_bundle(x_u, x_b)
        return preds.detach().numpy()


def train(u, i, j):
    # u, i, j = u.tolist(), i.tolist(), j.tolist()
    p_ui = model(u, i)
    p_uj = model(u, j)
    # loss = -torch.sum(torch.log(torch.sigmoid(p_ui - p_uj)))
    # loss = -torch.mean(torch.log(torch.sigmoid(p_ui - p_uj)))
    # loss = -F.logsigmoid(p_ui - p_uj).sum()
    loss = -F.logsigmoid(p_ui - p_uj).mean()
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    return loss


def train_bundle(u, i, j):
    # u, i, j = u.tolist(), i.tolist(), j.tolist()
    p_ui = model.forward_bundle(u, i)
    p_uj = model.forward_bundle(u, j)
    # loss = -torch.sum(torch.log(torch.sigmoid(p_ui - p_uj)))
    # loss = -torch.mean(torch.log(torch.sigmoid(p_ui - p_uj)))
    # loss = -F.logsigmoid(p_ui - p_uj).sum()
    loss = -F.logsigmoid(p_ui - p_uj).mean()
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    return loss


def rank_metrics(rank_data, batch_size, k):
    """Model performance on ranking metrics."""
    u_rank, p_rank, n_rank = rank_data
    assert len(u_rank) == len(p_rank) & len(u_rank) == len(n_rank)
    assert len(n_rank[0]) == 99
    n_user = len(u_rank)

    def _build_rank_data(u_rank, p_rank, n_rank):
        """用户-商品交互，正样本放在第一个位置，后面为99个负样本"""
        x_u = np.array([[u] * (1 + 99) for u in u_rank]).flatten()
        x_i = np.concatenate((p_rank.reshape(-1, 1), n_rank), axis=1).flatten()
        return x_u.reshape(-1, 1), x_i.reshape(-1, 1)

    if batch_size is None:
        x_u, x_i = _build_rank_data(u_rank, p_rank, n_rank)
        x_u, x_i = torch.tensor(x_u, dtype=torch.long), torch.tensor(x_i, dtype=torch.long)
        pred = model.predict_bundle(x_u, x_i)
    else:
        batch_gen = batch_generator_one_shot(rank_data, batch_size=batch_size, shuffle=False, drop_last=False)
        pred = []
        for u, p, n in batch_gen:
            x_u, x_i = _build_rank_data(u, p, n)
            x_u, x_i = torch.tensor(x_u, dtype=torch.long), torch.tensor(x_i, dtype=torch.long)
            tmp = model.predict_bundle(x_u, x_i)
            pred.append(tmp)
        pred = np.concatenate(pred, axis=0)
    pred = pred.reshape(n_user, -1)

    from utils.metrics import _hr_score, _mrr_score, _ndcg_score, _coverage

    hr = _hr_score(pred, k)
    mrr = _mrr_score(pred, k)
    ndcg = _ndcg_score(pred, k)
    # item = np.concatenate([p_rank.reshape(-1, 1), n_rank], axis=1)
    # print(pred.shape, item.shape)
    # cvg = _coverage(pred, item, k)

    if hr > 0.95:  # unusual metrics, e.g., the network output is all 0
        print(pred.shape, pred[0:3])
        print(np.mean(pred), np.std(pred), pred.sum(), np.all(pred == 0))

    return hr, mrr, ndcg


def run(train_mat, train_item, test_data, bundle_item, config):

    k = config['top_k']
    n_epochs = config['n_epochs']
    batch_size = config['batch_size']
    n_steps = int(train_mat.nnz * n_epochs / batch_size)
    eval_steps = int(train_mat.nnz * 1 / batch_size)
    print("Epochs: %d, Batch Size: %d, Samples: %d, Train Steps: %d, Evaluate Steps: %d"
          % (n_epochs, batch_size, train_mat.nnz, n_steps, eval_steps))

    n_user, n_item, n_bundle = config['n_user'], config['n_item'], config['n_bundle']

    model.eval()
    start = time.time()
    hr, mrr, ndcg = rank_metrics(test_data, batch_size=batch_size * 10, k=k)
    elapsed = time.time() - start
    log = 'Step: {:03d}, Time: {:.4f}, HR@{}: {:.4f}, MRR@{}: {:.4f}, NDCG@{}: {:.4f}'
    print(log.format(0, elapsed, k, hr, k, mrr, k, ndcg))

    best_epoch, best_hr, best_mrr, best_ndcg, = -1, hr, mrr, ndcg

    for step in range(1, n_steps + 1):
        model.train()
        start = time.time()
        for _ in range(1):
            u, i, j = sample_batch(train_item, n_user, n_item, batch_size)
            u, i, j = torch.tensor(u, dtype=torch.long), \
                      torch.tensor(i, dtype=torch.long), torch.tensor(j, dtype=torch.long)
            loss1 = train(u, i, j)
        u, i, j = sample_batch(train_mat, n_user, n_bundle, batch_size)
        u, i, j = torch.tensor(u, dtype=torch.long), \
                  torch.tensor(i, dtype=torch.long), torch.tensor(j, dtype=torch.long)
        loss2 = train_bundle(u, i, j)
        elapsed = time.time() - start
        log = '\tStep: {:03d}, Time: {:.4f}, Item Loss: {:.4f}, Bundle Loss: {:.4f}'
        print(log.format(step, elapsed, loss1.item(), loss2.item()))

        if step % eval_steps == 0:
            model.eval()
            start = time.time()
            hr, mrr, ndcg = rank_metrics(test_data, batch_size=batch_size * 10, k=k)
            elapsed = time.time() - start
            log = '\tStep: {:03d}, Time: {:.4f}, HR@{}: {:.4f}, MRR@{}: {:.4f}, NDCG@{}: {:.4f}'
            print(log.format(step, elapsed, k, hr, k, mrr, k, ndcg))

            if ndcg > best_ndcg:
                best_epoch, best_hr, best_mrr, best_ndcg = step, hr, mrr, ndcg

    return best_hr, best_mrr, best_ndcg


if __name__ == '__main__':
    
    device = torch.device("cpu")

    from bundle.load_data import load_movielens, load_data_steam, load_data_youshu, load_data_netease, load_data_yujian

    # load dataset
    data_dict = load_data_steam()
    n_user, n_item, n_bundle = data_dict['n_user'], data_dict['n_item'], data_dict['n_bundle']
    train_mat = data_dict['train_mat']  # train interaction matrix
    test_mat = data_dict['test_mat']  # test interaction matrix
    u_rank, p_rank = test_mat.nonzero()  # Positive user-item interaction
    n_rank = np.array(data_dict['negative'])  # Negative Sampling
    test_data = (u_rank, p_rank, n_rank)  # Triplet test data
    print(type(u_rank), type(p_rank), type(n_rank))
    print(u_rank.shape, p_rank.shape, n_rank.shape)

    train_item = data_dict['user_item']
    bundle_item = data_dict['bundle_item']

    # configuration
    config = dict()
    config['n_class'] = 2
    config['n_user'] = n_user
    config['n_item'] = n_item
    config['n_bundle'] = n_bundle
    config['embed_dim'] = 16
    config['n_epochs'] = 10
    config['batch_size'] = 1024
    config['top_k'] = 5

    # build the model
    model = DAM(config)

    # define loss function
    # loss_fn = F.binary_cross_entropy_with_logits
    # loss_fn = torch.nn.CrossEntropyLoss().to(device)
    def loss_fn(p_ui, p_uj):
        loss = -F.logsigmoid(p_ui - p_uj).sum()
        # loss = -(p_ui - p_uj).sigmoid().log().sum()
        return loss

    # define an optimizer
    # optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=1e-5)

    # train and evaluate the model
    hr, mrr, ndcg = run(train_mat, train_item, test_data, bundle_item, config)

    print('-' * 80)
    print('HR: {:.4f}, MRR: {:.4f}, NDCG: {:.4f}'.format(hr, mrr, ndcg))
    print('-' * 80)
