import os
import pickle

import numpy as np

import torch
from torch.utils.data import TensorDataset

from utils.data_utils import sample_triplet

data_dir = "/root/reclib/data"
# data_dir = "C:/Users/dengqilin/Desktop/code/reclib/data"
if not os.path.isdir(data_dir):
    os.mkdir(data_dir)


def load_movielens():
    path = os.path.join(data_dir, 'movielens', 'data.pickle')
    names = [
        'n_user', 'n_item', 'mat',
        'train_mat', 'test_mat', 'negative'
    ]
    with open(path, 'rb') as f:
        items = pickle.load(f)
    data_dict = dict(zip(names, items))
    return data_dict


def load_data_steam():
    path = os.path.join(data_dir, 'Steam', 'data.pickle')
    names = [
        'n_user', 'n_bundle', 'n_item', 'user_bundle', 'user_item', 'bundle_item',
        'train_mat', 'test_mat', 'negative', 'train_item', 'test_item', 'negative_items'
    ]
    with open(path, 'rb') as f:
        items = pickle.load(f)
    data_dict = dict(zip(names, items))
    return data_dict


def load_data_youshu():
    path = os.path.join(data_dir, 'Youshu', 'data.pickle')
    names = [
        'n_user', 'n_bundle', 'n_item', 'user_bundle', 'user_item', 'bundle_item',
        'train_mat', 'test_mat', 'negative', 'train_item', 'test_item'
    ]
    with open(path, 'rb') as f:
        items = pickle.load(f)
    data_dict = dict(zip(names, items))
    return data_dict


def load_data_netease():
    path = os.path.join(data_dir, 'NetEase', 'data.pickle')
    names = [
        'n_user', 'n_bundle', 'n_item', 'user_bundle', 'user_item', 'bundle_item',
        'train_mat', 'test_mat', 'negative'
    ]
    with open(path, 'rb') as f:
        items = pickle.load(f)
    data_dict = dict(zip(names, items))
    return data_dict


def load_data_yujian():
    path = os.path.join(data_dir, 'Yujian', 'data.pickle')
    names = [
        'n_user', 'n_bundle', 'n_item', 'user_bundle', 'user_item', 'bundle_item',
        'train_mat', 'test_mat', 'negative'
    ]
    with open(path, 'rb') as f:
        items = pickle.load(f)
    data_dict = dict(zip(names, items))
    return data_dict


def load_dataset(dataset='netease', bundle=True):
    if dataset == 'steam':
        data_dict = load_data_steam()
    elif dataset == 'youshu':
        data_dict = load_data_youshu()
    elif dataset == 'netease':
        data_dict = load_data_netease()
    elif dataset == 'yujian':
        data_dict = load_data_yujian()
    else:
        data_dict = None
        pass
    n_user, n_item, n_bundle = data_dict['n_user'], data_dict['n_item'], data_dict['n_bundle']
    train_mat = data_dict['train_mat']  # user-item interaction matrix
    u_train, p_train, n_train = sample_triplet(train_mat, n_user, n_bundle)
    test_mat = data_dict['test_mat']  # user-item interaction matrix
    u_rank, p_rank = test_mat.nonzero()  # Positive user-item interaction
    n_rank = np.array(data_dict['negative'])  # Negative Sampling
    # test_data = (u_rank, p_rank, n_rank)
    bundle_item = data_dict['bundle_item']
    u_train, p_train, n_train, u_rank, p_rank, n_rank = map(
        torch.tensor, (u_train, p_train, n_train, u_rank, p_rank, n_rank)
    )
    train_ds = TensorDataset(u_train.view(-1, 1), p_train.view(-1, 1), n_train.view(-1, 1))
    rank_ds = TensorDataset(u_rank.view(-1, 1), p_rank.view(-1, 1), n_rank)
    if bundle:
        return train_ds, rank_ds, n_user, n_item, n_bundle, train_mat, bundle_item
    else:
        user_item = data_dict['user_item']
        u_train, p_train, n_train = sample_triplet(user_item, n_user, n_item)
        u_train, p_train, n_train, = map(torch.tensor, (u_train, p_train, n_train))
        user_item_ds = TensorDataset(u_train.view(-1, 1), p_train.view(-1, 1), n_train.view(-1, 1))
        return user_item_ds, None, n_user, n_item, user_item
