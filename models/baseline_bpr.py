"""
https://github.com/gamboviol/bpr
https://github.com/guoyang9/BPR-pytorch
https://github.com/sh0416/bpr
"""

import os
import time
import random
import pickle
import argparse

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from torch.utils.data import TensorDataset, DataLoader
from torch.utils.data.dataset import random_split
from torch.utils.tensorboard import SummaryWriter


class BPR(nn.Module):

    def __init__(self, n_user, n_item, embed_dim):
        super(BPR, self).__init__()
        self.n_user = n_user
        self.n_item = n_item
        self.embed_user = nn.Embedding(n_user, embed_dim)
        self.embed_item = nn.Embedding(n_item, embed_dim)
        # nn.init.normal_(self.embed_user.weight, std=0.01)
        # nn.init.normal_(self.embed_item.weight, std=0.01)
        nn.init.xavier_normal_(self.embed_user.weight)
        nn.init.xavier_normal_(self.embed_item.weight)

    def forward(self, u, i, j):
        """
        Note that using (batch_size, 1) instead of (batch_size,)
        although (batch_size,) is ok because of squeeze compatibility
        :param u: (batch_size, 1)
        :param i: (batch_size, 1)
        :param j: (batch_size, 1)
        :return: (batch_size, 1)
        """
        # print(u.shape, i.shape, j.shape)
        u = self.embed_user(u.long()).squeeze(dim=1)  # (batch_size, embed_dim)
        i = self.embed_item(i.long()).squeeze(dim=1)
        j = self.embed_item(j.long()).squeeze(dim=1)
        p_ui = torch.mul(u, i).sum(dim=-1, keepdim=True)  # (batch_size, 1)
        p_uj = torch.mul(u, j).sum(dim=-1, keepdim=True)
        # p_ui = (u * i).sum(dim=-1)
        # p_uj = (u * j).sum(dim=-1)
        # print(p_ui.shape, p_uj.shape)
        return p_ui, p_uj

    def predict(self, u, i):
        # print(u.shape, i.shape)
        u = self.embed_user(u.long()).squeeze(dim=1)  # (batch_size, 1, embed_dim)
        i = self.embed_item(i.long()).squeeze(dim=1)
        p_ui = torch.mul(u, i).sum(dim=-1, keepdim=True)  # (batch_size, 1)
        # p_ui = (u * i).sum(dim=-1)
        # print(p_ui.shape)
        return p_ui

    def recommend(self, u):
        u = self.embed_user(u.long()).squeeze(dim=1)   # (batch_size, embed_dim)
        p_ui = torch.mm(u, self.embed_item.weight.t())  # (batch_size, n_item)
        pred = torch.argsort(p_ui, dim=1)
        return pred

    def func(self):
        pass


if __name__ == '__main__':

    writer = SummaryWriter()

    device = torch.device("cpu")
    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    from bundle.load_data import load_movielens

    # load dataset
    train_ds, rank_ds, n_user, n_item, train_mat = load_movielens()
    train_len = int(len(train_ds) * 0.95)
    train_ds, valid_ds = random_split(train_ds, [train_len, len(train_ds) - train_len])

    # configuration
    EMBED_DIM = 32
    LR = 0.001
    LAMBDA = 1e-5
    N_EPOCHS = 5
    BATCH_SIZE = 128
    TOP_K = 5

    config = dict()
    config['device'] = device

    # build the model
    model = BPR(n_user, n_item, EMBED_DIM).to(device)

    # define loss function
    # loss_fn = torch.nn.CrossEntropyLoss().to(device)
    def loss_fn(p_ui, p_uj):
        loss = -F.logsigmoid(p_ui - p_uj).sum()
        # loss = -(p_ui - p_uj).sigmoid().log().sum()
        return loss

    # define an optimizer
    optimizer = optim.Adam(model.parameters(), lr=LR, weight_decay=LAMBDA)
    # optimizer = torch.optim.SGD(model.parameters(), lr=4.0)
    # scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 1, gamma=0.9)

    # # train and evaluate the model
    # fit(model, train_ds, valid_ds, loss_fn, optimizer, N_EPOCHS, BATCH_SIZE)
    # # model inference
    # predict(model, test_ds, BATCH_SIZE * 4)

    from utils.pipeline import PipelineBPR as Pipeline

    # train and evaluate the model
    pl = Pipeline(model, config)
    pl.compile(optimizer, loss_fn, metrics=None)
    # pl.fit(train_ds, valid_ds, BATCH_SIZE, N_EPOCHS)
    # pl.fit(train_ds, valid_ds, BATCH_SIZE, N_EPOCHS, rank_ds, TOP_K)
    # model inference
    # pl.predict(test_ds, BATCH_SIZE * 4)

    def _resample_triplet():
        from utils.data_utils import sample_triplet
        u_train, p_train, n_train = sample_triplet(train_mat, n_user, n_item)
        u_train, p_train, n_train = map(torch.tensor, (u_train, p_train, n_train))
        train_ds = TensorDataset(u_train, p_train, n_train)
        return train_ds

    for i in range(N_EPOCHS):
        train_ds = _resample_triplet()
        hr, mrr, ndcg = pl.fit(train_ds, valid_ds, BATCH_SIZE, 1, rank_ds, TOP_K)

        print('-' * 80)
        print('HR: {:.4f}, MRR: {:.4f}, NDCG: {:.4f}'.format(hr, mrr, ndcg))
        print('-' * 80)

    # https://pytorch.org/docs/stable/notes/serialization.html
    # https://pytorch.org/tutorials/beginner/saving_loading_models.html
    model_path = './model_bpr.pth'
    torch.save(model.state_dict(), model_path)
    model = BPR(n_user, n_item, EMBED_DIM).to(device)
    model.load_state_dict(torch.load(model_path))
    # model_path = './model_bpr.entire.pth'
    # torch.save(model, model_path)
    # model = torch.load(model_path)
