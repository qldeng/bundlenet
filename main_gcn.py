
import numpy as np

import torch
import torch.nn as nn
from torch.nn import Parameter
import torch.nn.functional as F
import torch.optim as optim

from utils.pipeline import Pipeline
from utils.load_data import load_movielens, load_data_steam, load_data_youshu, load_data_netease, load_data_yujian
from models.baseline_gcn import GCN, build_graph_data, run

if __name__ == '__main__':

    device = torch.device("cpu")

    for dataset in ['steam', 'youshu',  'netease', 'yujian']:

        if dataset == 'steam':
            data_dict = load_data_steam()
        elif dataset == 'youshu':
            data_dict = load_data_youshu()
        elif dataset == 'netease':
            data_dict = load_data_netease()
        else:
            data_dict = load_data_yujian()

        # load dataset
        n_user, n_bundle = data_dict['n_user'], data_dict['n_bundle']
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
        config['n_item'] = n_bundle
        config['n_node'] = n_user + n_bundle
        config['embed_dim'] = 32
        config['mlp'] = False
        config['triplet'] = False  # using triplet loss
        config['n_epochs'] = 20
        config['batch_size'] = None
        config['device'] = 'cpu'
        config['top_k'] = 5
        config['lr'] = 1e-5  # learning rate
        config['lambda'] = 1e-5  # regularization coefficient

        # global graph data
        data = build_graph_data(train_mat, config)
        print(data)

        # build the model
        model = GCN(config, data).to(config['device'])

        # define loss function
        loss_fn = F.binary_cross_entropy_with_logits

        # define an optimizer
        # optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=1e-5)

        # train and evaluate the model
        pl = Pipeline(model, config)
        pl.compile(optimizer, loss_fn, metrics=None)
        hr, mrr, ndcg = run(pl, train_mat, test_data, config)

        print('-' * 80)
        print('The metrics for dataset ({}) is: '.format(dataset))
        print('HR: {:.4f}, MRR: {:.4f}, NDCG: {:.4f}'.format(hr, mrr, ndcg))
        print('-' * 80)
