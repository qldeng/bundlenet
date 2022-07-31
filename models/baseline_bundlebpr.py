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
from torch.nn.parameter import Parameter
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from torch.utils.data.dataset import random_split
from torch.utils.tensorboard import SummaryWriter

from utils.pipeline import PipelineBPR as Pipeline

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# writer = SummaryWriter()


class BundleBPR(nn.Module):

    def __init__(self, embed_dim, n_user, n_item, n_bundle, mat, max_bundle_size):
        super(BundleBPR, self).__init__()
        self.n_user = n_user
        self.n_item = n_item
        self.n_bundle = n_bundle
        self.mat = mat
        self.max_bundle_size = max_bundle_size
        self.embed_dim = embed_dim
        self.embed_user = nn.Embedding(n_user, embed_dim)
        self.embed_item = nn.Embedding(n_item, embed_dim)
        # print(list(self.parameters()))
        for p in self.parameters():
            p.requires_grad = False
        # self.embed_bundle = nn.Embedding(n_bundle, embed_dim)
        self.w1 = Parameter(torch.Tensor(embed_dim, embed_dim))
        self.w2 = Parameter(torch.Tensor(embed_dim, embed_dim))
        # self.bias = Parameter(torch.Tensor(n_item))

    def forward(self, u, i, j):
        """
        Note that using (batch_size, 1) instead of (batch_size,)
        although (batch_size,) is ok because of squeeze compatibility
        :param u: (batch_size, 1)
        :param i: (batch_size, 1)
        :param j: (batch_size, 1)
        :return: (batch_size, 1)
        """
        embed_u = self.embed_user(u.long()).squeeze(dim=1)  # (batch_size, embed_dim)

        # print(i, type(i), i.shape)
        i = i.cpu()
        items = []
        for k in i:
            # print(k, type(k), k.shape)
            tmp = self.mat[k, :].nonzero()[1].tolist()[0:self.max_bundle_size]
            tmp = tmp + [-1] * (self.max_bundle_size - len(tmp))
            items.append(tmp)
        i = torch.tensor(items).to(device)

        mask_i = torch.gt(i, -1)
        i[i.eq(-1)] = 0
        mask_i = mask_i.unsqueeze(2).expand(-1, -1, self.embed_dim)
        embed_i = self.embed_item(i)
        # embed_i = self.embed_item(i.long())
        embed_i = embed_i * mask_i.to(dtype=torch.float)
        embed_i = embed_i.mean(dim=1)  # (batch_size, embed_dim)

        # split_size = i.gt(0).sum(dim=1).tolist()
        # tensors = self.embed_item(i)[mask_i].view(-1, self.embed_dim).split(split_size, dim=0)
        # tensors = [torch.mean(t, dim=0, keepdim=True) for t in tensors]
        # embed_i = torch.cat(tensors, dim=0)

        j = j.cpu()
        items = []
        for k in j:
            tmp = self.mat[k, :].nonzero()[1].tolist()[0:self.max_bundle_size]
            tmp = tmp + [-1] * (self.max_bundle_size - len(tmp))
            items.append(tmp)
        j = torch.tensor(items).to(device)

        mask_j = torch.gt(j, -1)
        j[j.eq(-1)] = 0
        mask_j = mask_j.unsqueeze(2).expand(-1, -1, self.embed_dim)
        embed_j = self.embed_item(j)
        embed_j = embed_j * mask_j.to(dtype=torch.float)
        embed_j = embed_j.mean(dim=1)  # (batch_size, embed_dim)

        h_u = embed_u.matmul(self.w1.t())  # (batch_size, embed_dim)
        h_i = embed_i.matmul(self.w2.t())  # (batch_size, embed_dim)
        h_j = embed_j.matmul(self.w2.t())  # (batch_size, embed_dim)

        p_ui = torch.mul(h_u, h_i).sum(dim=-1, keepdim=True)  # (batch_size, 1)
        p_uj = torch.mul(h_u, h_j).sum(dim=-1, keepdim=True)  # (batch_size, 1)
        # p_ui = (h_u * h_i).sum(dim=-1)
        # p_uj = (h_u * h_j).sum(dim=-1)
        return p_ui, p_uj

    def predict(self, u, i):
        embed_u = self.embed_user(u.long()).squeeze(dim=1)  # (batch_size, embed_dim)

        i = i.cpu()
        items = []
        for k in i:
            tmp = self.mat[k, :].nonzero()[1].tolist()[0:self.max_bundle_size]
            tmp = tmp + [-1] * (self.max_bundle_size - len(tmp))
            items.append(tmp)
        i = torch.tensor(items).to(device)

        mask_i = torch.gt(i, -1)
        i[i.eq(-1)] = 0
        mask_i = mask_i.unsqueeze(2).expand(-1, -1, self.embed_dim)
        embed_i = self.embed_item(i)
        # embed_i = self.embed_item(i.long())
        embed_i = embed_i * mask_i.to(dtype=torch.float)
        embed_i = embed_i.mean(dim=1)  # (batch_size, embed_dim)

        p_ui = torch.mul(embed_u, embed_i).sum(dim=-1, keepdim=True)  # (batch_size, 1)
        # p_ui = (u * i).sum(dim=-1)
        return p_ui

    def recommend(self, u):
        u = self.embed_user(u.long()).squeeze(dim=1)  # (batch_size, embed_dim)
        p_ui = torch.mm(u, self.embed_item.weight.t())  # (batch_size, n_item)
        pred = torch.argsort(p_ui, dim=1)
        return pred

    def func(self):
        pass


EMBED_DIM = 32

LR = 0.001
LAMBDA = 1e-5

N_EPOCHS = 5
BATCH_SIZE = 128
TOP_K = 10
MAX_BUNDLE_SIZE = 100

# load dataset
from bundle.load_data import load_data_netease

train_ds, rank_ds, n_user, n_item, n_bundle, train_mat, bundle_item = load_data_netease(bundle=True)
train_len = int(len(train_ds) * 0.95)
train_ds, valid_ds = random_split(train_ds, [train_len, len(train_ds) - train_len])

# define the model
model = BundleBPR(EMBED_DIM, n_user, n_item, n_bundle, bundle_item, MAX_BUNDLE_SIZE).to(device)


# define loss function
# loss_fn = torch.nn.CrossEntropyLoss().to(device)
def loss_fn(p_ui, p_uj):
    loss = -F.logsigmoid(p_ui - p_uj).sum()
    # loss = -(p_ui - p_uj).sigmoid().log().sum()
    return loss


# define an optimizer
optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=LR, weight_decay=LAMBDA)
# optimizer = optim.Adam(model.parameters(), lr=LR, weight_decay=LAMBDA)
# optimizer = torch.optim.SGD(model.parameters(), lr=4.0)
# scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 1, gamma=0.9)

# # train and evaluate the model
# fit(model, train_ds, valid_ds, loss_fn, optimizer, N_EPOCHS, BATCH_SIZE)
# # model inference
# predict(model, test_ds, BATCH_SIZE * 4)

m = Pipeline(model)
m.compile(optimizer, loss_fn, metrics=None)

PATH = './model_bpr.pth'
model.load_state_dict(torch.load(PATH), strict=False)

# train and evaluate the model
# m.fit(train_ds, valid_ds, BATCH_SIZE, N_EPOCHS)
# m.fit(train_ds, valid_ds, BATCH_SIZE, N_EPOCHS, rank_ds, TOP_K)

for i in range(N_EPOCHS):

    def _resample_triplet():
        from utils.data_utils import sample_triplet
        u_train, p_train, n_train = sample_triplet(train_mat, n_user, n_bundle)
        u_train, p_train, n_train = map(torch.tensor, (u_train, p_train, n_train))
        train_ds = TensorDataset(u_train, p_train, n_train)
        return train_ds

    train_ds = _resample_triplet()
    m.fit(train_ds, valid_ds, BATCH_SIZE, 1, rank_ds, TOP_K)

PATH = './model_bundlebpr.pth'
torch.save(model.state_dict(), PATH)

# model inference
# m.predict(test_ds, BATCH_SIZE * 4)
