"""
https://keras.io/api/metrics/
https://scikit-learn.org/stable/modules/classes.html#module-sklearn.metrics
https://medium.com/swlh/rank-aware-recsys-evaluation-metrics-5191bba16832
https://www.jianshu.com/p/665f9f168eff
"""

import math
import heapq
import sys
import copy
import random

import numpy as np
import pandas as pd

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
from torch.utils.data import TensorDataset

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.metrics import dcg_score, ndcg_score


def to_array(*args, **kwargs):
    """
    https://pytorch.org/docs/stable/tensors.html#torch.Tensor.numpy
    https://scikit-learn.org/stable/modules/generated/sklearn.utils.validation.column_or_1d.html
    """
    a = []
    for v in args:
        if not isinstance(v, np.ndarray):
            v = np.asarray(v, **kwargs)
        a.append(v)
    return tuple(a)


def to_tensor(*args, **kwargs):
    """

    https://pytorch.org/docs/stable/generated/torch.as_tensor.html
    https://pytorch.org/docs/stable/generated/torch.from_numpy.html
    """
    a = []
    for v in args:
        if not isinstance(v, torch.Tensor):
            v = torch.Tensor(v, **kwargs)
        a.append(v)
    return tuple(a)


def as_tensor(*args, **kwargs):
    a = []
    for v in args:
        v = torch.as_tensor(v, **kwargs)
        a.append(v)
    return tuple(a)


def _sigmoid(x):
    return 1 / (1 + np.exp(-x))


def _softmax(x):
    return np.exp(x) / np.sum(np.exp(x))


def ordinary_accuracy(y_true, y_pred):
    y_true, y_pred = to_array(y_true, y_pred)
    assert y_true.shape == y_pred.shape
    count = (y_true == y_pred).sum()
    total = y_true.shape[0]
    return count / total


def binary_accuracy(y_true, y_pred, threshold=0.5, sigmoid=True):
    y_true, y_pred = to_array(y_true, y_pred)
    assert y_true.shape[0] == y_pred.shape[0]
    if sigmoid:
        y_pred = _sigmoid(y_pred)
    # y_pred = np.round(y_pred)
    # y_pred = np.greater(y_pred, threshold)
    y_pred = (y_pred > threshold).astype(int)
    count = (y_true == y_pred).sum()
    total = y_true.shape[0]
    return count / total


def categorical_accuracy(y_true, y_pred, softmax=False):
    y_true, y_pred = to_array(y_true, y_pred)
    assert y_true.shape[0] == y_pred.shape[0]
    if softmax:
        y_pred = _softmax(y_pred)
    y_pred = np.argmax(y_pred, axis=1)
    count = (y_true == y_pred).sum()
    total = y_true.shape[0]
    return count / total


def torch_ordinary_accuracy(y_true, y_pred):
    """ordinary accuracy."""
    assert y_true.size(0) == y_pred.size(0)
    count = torch.eq(y_pred.type(y_true.type()), y_true).sum().item()
    total = y_true.size(0)
    return count / total


def torch_binary_accuracy(y_true, y_pred, threshold=0.5, sigmoid=True):
    """binary accuracy."""
    assert y_true.size(0) == y_pred.size(0)
    if sigmoid:
        y_pred = torch.sigmoid(y_pred)
    # y_pred = torch.round(y_pred)
    # y_pred = torch.gt(y_pred, threshold)
    y_pred = y_pred > threshold
    count = torch.eq(y_pred.type(y_true.type()), y_true).sum().item()
    total = y_true.size(0)
    return count / total


def torch_categorical_accuracy(y_true, y_pred, softmax=False):
    """categorical accuracy."""
    assert y_true.size(0) == y_pred.size(0)
    if softmax:
        y_pred = F.softmax(y_pred, dim=1)
    y_pred = y_pred.argmax(dim=1)
    count = torch.eq(y_pred.type(y_true.type()), y_true).sum().item()
    total = y_true.size(0)
    return count / total


def accuracy(y_true, y_pred):
    accuracy = accuracy_score(y_true, y_pred)
    return accuracy


def precision(y_true, y_pred):
    precision = precision_score(y_true, y_pred, average='binary')
    return precision


def recall(y_true, y_pred):
    recall = recall_score(y_true, y_pred, average='binary')
    return recall


def roc_auc(y_true, y_pred):
    roc_auc = roc_auc_score(y_true, y_pred)
    return roc_auc


def metrics(y_true, y_pred):
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, pos_label=1, average='binary')
    recall = recall_score(y_true, y_pred, pos_label=1, average='binary')
    f1 = f1_score(y_true, y_pred, pos_label=1, average='binary')
    roc_auc = roc_auc_score(y_true, y_pred)
    return accuracy, precision, recall, f1, roc_auc


def mae(y_true, y_pred):
    mae = mean_absolute_error(y_true, y_pred)
    # mae = np.average(np.abs(np.array(y_true) - np.array(y_pred)))
    return mae


def mse(y_true, y_pred, squared=True):
    mse = mean_squared_error(y_true, y_pred)
    # mse = np.average((np.array(y_true) - np.array(y_pred)) ** 2)
    return mse if squared else np.sqrt(mse)


def dcg(y_true, y_pred, k=None):
    dcg = dcg_score(y_true, y_pred, k=k)
    return dcg


def ndcg(y_true, y_pred, k=None):
    ndcg = ndcg_score(y_true, y_pred, k=k)
    return ndcg


def hr_(y_true, y_pred, k=None):
    """Hit Ratio.
    https://numpy.org/doc/stable/reference/generated/numpy.argsort.html
    https://numpy.org/doc/stable/reference/generated/numpy.take_along_axis.html
    :param y_true: ndarray, shape (n_samples, n_labels)
    :param y_pred: ndarray, shape (n_samples, n_labels)
    :param k: int, optional (default=None)
    :return: float score
    """
    if k is None:
        indices = (-y_pred).argsort(axis=1)[:]
    else:
        indices = (-y_pred).argsort(axis=1)[:, :k]
    result = np.take_along_axis(y_true, indices, axis=1)  # (n_samples, k)
    hr = np.mean(np.sum(result, axis=1))
    return hr


def mrr_(y_true, y_pred, k=None):
    """Mean Reciprocal Rank.
    https://en.wikipedia.org/wiki/Mean_reciprocal_rank
    https://stackoverflow.com/questions/4588628/find-indices-of-elements-equal-to-zero-in-a-numpy-array
    :param y_true: ndarray, shape (n_samples, n_labels)
    :param y_pred: ndarray, shape (n_samples, n_labels)
    :param k: int, optional (default=None)
    :return: float score
    """
    if k is None:
        indices = (-y_pred).argsort(axis=1)[:]
    else:
        indices = (-y_pred).argsort()[:, :k]
    result = np.take_along_axis(y_true, indices, axis=1)  # (n_samples, k)
    mrr = np.mean(1.0 / (np.nonzero(result)[1] + 1))
    return mrr


def map_(y_true, y_pred, k=None):
    """Mean Average Precision.
    https://en.wikipedia.org/wiki/Evaluation_measures_(information_retrieval)#Mean_average_precision
    https://xgboost.readthedocs.io/en/latest/parameter.html#learning-task-parameters
    """
    pass


def ndcg_(y_true, y_pred, k=None):
    """Normalized Discounted Cumulative Gain.
    https://en.wikipedia.org/wiki/Discounted_cumulative_gain
    https://scikit-learn.org/stable/modules/generated/sklearn.metrics.ndcg_score.html
    :param y_true: ndarray, shape (n_samples, n_labels)
    :param y_pred: ndarray, shape (n_samples, n_labels)
    :param k: int, optional (default=None)
    :return: float score
    """
    ndcg = ndcg_score(y_true, y_pred, k=k)
    return ndcg


def _hr(pred, k):
    cnt = 0
    indices = (-pred).argsort()[:, :k]
    for i in range(len(indices)):
        if 0 in indices[i]:
            cnt += 1.
    return cnt / len(indices)


def _mrr(pred, k):
    rr = 0
    indices = (-pred).argsort()[:, :k]
    for i in range(len(indices)):
        for rank, idx in enumerate(indices[i]):
            if idx == 0:
                rr += 1.0 / (rank + 1)
                break
    return rr / len(indices)


def _ndcg(pred, k):
    ndcg_list = []
    indices = (-pred).argsort()[:, :k]
    for i in range(len(indices)):
        dcg = 0
        for rank, idx in enumerate(indices[i]):
            if idx == 0:
                dcg += 1. / np.log2(rank + 2)
                break
        idcg = 1. / np.log2(2)
        ndcg = dcg / idcg
        ndcg_list.append(ndcg)
    return np.mean(ndcg_list)


def _coverage(pred, item, k):
    """覆盖率
    测试集用户推荐出来的商品占训练集所有商品的比例
    """
    assert pred.shape == item.shape
    indices = (-pred).argsort()[:, :k]
    top_item = []
    for i in range(len(indices)):
        top_item.append(item[i][indices[i]])
    top_item = np.concatenate(top_item, axis=0)
    unique_top_item = np.unique(top_item.flatten())
    unique_item = np.unique(item.flatten())
    return len(unique_top_item) / len(unique_item)


def rank_metrics(pipeline, rank_data, batch_size, k):

    u_rank, p_rank, n_rank = rank_data[:]
    assert all(u_rank.size(0) == tensor.size(0) for tensor in (u_rank, p_rank, n_rank))
    assert len(u_rank) == len(p_rank) & len(u_rank) == len(n_rank)

    n_user = u_rank.size(0)
    n_negative = n_rank[0].size(0)

    u_rank = u_rank.expand(-1, 1 + n_negative)  # (n_user, (1 + n_negative))
    i_rank = torch.cat([p_rank.view(-1, 1), n_rank.int()], dim=1)  # (n_user, (1 + n_negative))

    y_pos = torch.ones_like(p_rank)
    y_neg = torch.zeros_like(n_rank)
    y_true = torch.cat([y_pos.view(-1, 1), y_neg.int()], dim=1)  # (n_user, (1 + n_negative))

    # u_rank = u_rank.view(-1, 1)  # (n_user * (1 + n_negative), 1)
    # u_rank = u_rank.contiguous().view(-1, 1)  # (n_user * (1 + n_negative), 1)
    u_rank = u_rank.reshape(-1, 1)  # (n_user * (1 + n_negative), 1)
    i_rank = i_rank.reshape(-1, 1)  # (n_user * (1 + n_negative), 1)
    dataset = TensorDataset(u_rank, i_rank)

    if batch_size is None:
        pred = pipeline.predict_batch(dataset.tensors)  # (n_user * (1 + n_negative), 1)
    else:
        pred = pipeline.predict(dataset, batch_size)  # (n_user * (1 + n_negative), 1)

    y_pred = pred.view(n_user, -1)  # (n_user, (1 + n_negative))

    hr = hr_(y_true.numpy(), y_pred.numpy(), k=k)
    mrr = mrr_(y_true.numpy(), y_pred.numpy(), k=k)
    ndcg = ndcg_(y_true.numpy(), y_pred.numpy(), k=k)

    return hr, mrr, ndcg


def _build_tuple(u_rank, p_rank, n_rank):
    """用户-商品交互，正样本放在第一个位置，后面为99个负样本"""
    x_u = np.array([[u] * (1 + 99) for u in u_rank]).flatten()
    x_i = np.concatenate((p_rank.reshape(-1, 1), n_rank), axis=1).flatten()
    return x_u.reshape(-1, 1), x_i.reshape(-1, 1)


def rank_metrics_v0(pipeline, rank_data, batch_size, k=10):

    u_rank, p_rank, n_rank = rank_data
    assert len(u_rank) == len(p_rank) & len(u_rank) == len(n_rank)
    assert len(n_rank[0]) == 99

    n_user = len(u_rank)

    x_u, x_i = _build_tuple(u_rank, p_rank, n_rank)
    x_u, x_i = map(torch.tensor, (x_u, x_i))
    dataset = TensorDataset(x_u.view(-1, 1), x_i.view(-1, 1))

    if batch_size is None:
        # https://pytorch.org/docs/stable/data.html#torch.utils.data.Subset
        pred = pipeline.predict_batch(dataset.tensors)
    else:
        pred = pipeline.predict(dataset, batch_size)

    pred = pred.reshape(n_user, -1)

    hr = _hr(pred, k)
    mrr = _mrr(pred, k)
    ndcg = _ndcg(pred, k)

    return hr, mrr, ndcg


def rank_metrics_v1(model, rank_data, batch_size=None, k=10):
    """Model performance on ranking metrics.
    if isinstance(pred, (list, tuple)):
        pred = np.array(pred)
    elif not isinstance(pred, np.ndarray):
        pred = np.array(pred)
    else:
        pass
    """
    u_rank, p_rank, n_rank = rank_data
    assert len(u_rank) == len(p_rank) & len(u_rank) == len(n_rank)
    assert len(n_rank[0]) == 99

    n_user = len(u_rank)

    x_u, x_i = _build_tuple(u_rank, p_rank, n_rank)

    pred = model.predict((x_u, x_i))

    pred = pred.reshape(n_user, -1)

    hr = _hr(pred, k)
    mrr = _mrr(pred, k)
    ndcg = _ndcg(pred, k)

    # item = np.concatenate([p_rank.reshape(-1, 1), n_rank], axis=1)
    # cvg = _coverage(pred, item, k)

    return hr, mrr, ndcg


def rank_metrics_v1_(model, rank_data, batch_size=1024, k=10):
    """Evaluate model performance."""
    u_rank, p_rank, n_rank = rank_data
    assert len(u_rank) == len(p_rank) & len(u_rank) == len(n_rank)
    from utils.data_utils import batch_generator_one_shot
    batch_gen = batch_generator_one_shot(rank_data, batch_size=batch_size, shuffle=False, drop_last=False)
    pred = []
    for u_rank, p_rank, n_rank in batch_gen:
        x_u, x_i = _build_tuple(u_rank, p_rank, n_rank)
        tmp = model.predict((x_u, x_i), batch_size=1024)
        pred.append(tmp)
    pred = np.concatenate(pred, axis=0)
    pred = pred.reshape(len(u_rank), -1)
    hr = _hr(pred, k)
    mrr = _mrr(pred, k)
    ndcg = _ndcg(pred, k)
    return hr, mrr, ndcg


def rank_metrics_v2(model, rank_data, k=10):
    """ranking metrics."""
    users, p_items, n_items = rank_data
    assert len(users) == len(p_items) & len(users) == len(n_items)

    def _rank(model, u, item_list, k):
        """Ranking-based Recommendation."""
        u_rank = np.array([u] * len(item_list))
        # u_rank = np.full(len(item_list), u)
        i_rank = np.array(item_list)
        pred = model.predict([u_rank, i_rank], batch_size=1024)
        rank_score = {}
        for i in range(len(item_list)):
            item = item_list[i]
            rank_score[item] = pred[i]
        rank_list = heapq.nlargest(k, rank_score, key=rank_score.get)
        return rank_list

    def _hr(item, rank_list):
        """Hit Ratio."""
        if item in rank_list:
            return 1
        return 0.0

    def _mrr(item, rank_list):
        """Mean Reciprocal Rank."""
        for i, rank_item in enumerate(rank_list):
            if rank_item == item:
                return 1 / (i + 1)
        return 0.0

    def _ndcg(item, rank_list):
        """Normalized Discounted Cumulative Gain."""
        for i, rank_item in enumerate(rank_list):
            if rank_item == item:
                return 1 / math.log2(i + 1 + 1)
        return 0.0

    hr_list = []
    mrr_list = []
    ndcg_list = []
    for i in range(len(users)):
        u, p, n = users[i], p_items[i], n_items[i]
        item_list = [p]
        item_list.extend(n)
        rank_list = _rank(model, u, item_list, k)
        hr = _hr(p, rank_list)
        mrr = _mrr(p, rank_list)
        ndcg = _ndcg(p, rank_list)
        hr_list.append(hr)
        mrr_list.append(mrr)
        ndcg_list.append(ndcg)
    hr = np.array(hr_list).mean()
    mrr = np.array(mrr_list).mean()
    ndcg = np.array(ndcg_list).mean()
    return hr, mrr, ndcg
