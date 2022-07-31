import time
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from torch.utils.data import TensorDataset, DataLoader
from torch.utils.data.dataset import random_split
from torch.utils.tensorboard import SummaryWriter

from models.baseline_bpr import BPR
from utils.load_data import load_movielens, load_dataset
from utils.pipeline import PipelineBPR as Pipeline


if __name__ == '__main__':

    device = torch.device("cpu")
    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    for dataset in ['netease']:
    # for dataset in ['steam', 'youshu',  'netease', 'yujian']:

        # # loading / building dataset
        # train_ds, rank_ds, n_user, n_item, train_mat = load_movielens()
        # train_len = int(len(train_ds) * 0.95)
        # train_ds, valid_ds = random_split(train_ds, [train_len, len(train_ds) - train_len])

        # load dataset
        train_ds, rank_ds, n_user, n_item, n_bundle, train_mat, bundle_item = load_dataset(dataset, bundle=True)
        train_len = int(len(train_ds) * 0.90)
        train_ds, valid_ds = random_split(train_ds, [train_len, len(train_ds) - train_len])

        # configuration
        EMBED_DIM = 32
        LR = 0.001
        LAMBDA = 1e-5
        N_EPOCHS = 10
        BATCH_SIZE = 1024
        TOP_K = 5

        config = dict()
        config['device'] = device

        # build the model
        model = BPR(n_user, n_bundle, EMBED_DIM).to(device)

        # define loss function
        # loss_fn = torch.nn.CrossEntropyLoss().to(device)
        def loss_fn(p_ui, p_uj):
            loss = -F.logsigmoid(p_ui - p_uj).sum()
            # loss = -(p_ui - p_uj).sigmoid().log().sum()
            return loss

        # define an optimizer
        optimizer = optim.Adam(model.parameters(), lr=LR, weight_decay=LAMBDA)

        # train and evaluate the model
        pl = Pipeline(model, config)
        pl.compile(optimizer, loss_fn, metrics=None)
        # pl.fit(train_ds, valid_ds, BATCH_SIZE, N_EPOCHS)
        # pl.fit(train_ds, valid_ds, BATCH_SIZE, N_EPOCHS, rank_ds, TOP_K)

        def _resample_triplet():
            from utils import sampling_triplet
            u_train, p_train, n_train = sampling_triplet(train_mat, n_user, n_bundle)
            u_train, p_train, n_train = map(torch.tensor, (u_train, p_train, n_train))
            train_ds = TensorDataset(u_train, p_train, n_train)
            return train_ds

        for i in range(N_EPOCHS):
            train_ds = _resample_triplet()
            hr, mrr, ndcg = pl.fit(train_ds, valid_ds, BATCH_SIZE, 1, rank_ds, TOP_K)

            print(f'\tHR: {hr:.4f}\t|\tMRR: {mrr:.4f}\t|\tNDCG: {ndcg:.4f}')
            # print('HR: {:.4f}, MRR: {:.4f}, NDCG: {:.4f}'.format(hr, mrr, ndcg))

        print('-' * 80)
        print('The metrics for dataset ({}) is: '.format(dataset))
        print('HR: {:.4f}, MRR: {:.4f}, NDCG: {:.4f}'.format(hr, mrr, ndcg))
        print('-' * 80)
