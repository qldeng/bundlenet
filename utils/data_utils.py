import random
from collections import defaultdict

import numpy as np
import scipy.sparse as sp
from scipy.sparse import dok_matrix, coo_matrix, csr_matrix
import pandas as pd


def build_vocab_df(df, start_idx=0, min_freq=1):
    """
    from utils.data_utils import build_vocab_df
    df, vocab = build_vocab_df(df, min_freq=1)
    df['user_id'] = df['user_id'].map(vocab['user2id'])
    df['item_id'] = df['item_id'].map(vocab['item2id'])
    """

    vocab = {}

    user_set = set(df['user_id'].unique().tolist())
    # user_freq = df['user_id'].value_counts().reset_index()
    user_freq = df['user_id'].value_counts().to_dict()
    user_set = sorted(filter(lambda v: user_freq[v] >= min_freq, user_set))
    n_user = len(user_set)
    user2id = dict(zip(user_set, range(start_idx, start_idx + len(user_set))))
    id2user = {user2id[k]: k for k in user2id}

    vocab['user2id'] = user2id
    vocab['id2user'] = id2user
    vocab['n_user'] = n_user

    item_set = set(df['item_id'].unique().tolist())
    # item_freq = df['item_id'].value_counts().reset_index()
    item_freq = df['item_id'].value_counts().to_dict()
    item_set = sorted(filter(lambda v: item_freq[v] >= min_freq, item_set))
    n_item = len(item_set)
    item2id = dict(zip(item_set, range(start_idx, start_idx + len(item_set))))
    id2item = {item2id[k]: k for k in item2id}

    vocab['item2id'] = item2id
    vocab['id2item'] = id2item
    vocab['n_item'] = n_item

    return df, vocab


def df2seq(df, vocab, implicit=True):
    """Transform pandas DataFrame to scipy csr_matrix."""

    assert 'ts' in df.columns
    # df = df.sort_values(by='ts', ascending=True)
    df = df.sort_values(by=['user_id', 'ts'], ascending=True)

    n_user, n_item = vocab['n_user'], vocab['n_item']
    if n_user is None or n_item is None:
        n_user = len(df['user_id'].unique())
        n_item = len(df['item_id'].unique())

    seq = defaultdict(list)

    for uid, iid, label, ts in df.values:
        label = 1 if implicit else label
        seq[uid].append(iid)

    # for i in range(len(df)):
    #     uid, iid, label = df.iloc[i][['user_id', 'item_id', 'label']]
    #     label = 1 if implicit else label
    #     seq[uid].append(iid)
    #     if (i + 1) % 10000 == 0:
    #         print("processed %d lines." % (i + 1))

    total_len = 0.0
    for u in seq:
        total_len += len(seq[u])
    avg_len = total_len / len(seq)
    print('average sequence length: %.2f' % avg_len)

    return seq


def split_dataset_seq(seq):

    train_ds = {}
    valid_ds = {}
    test_ds = {}

    for u in seq:
        if len(seq[u]) < 3:
            train_ds[u] = seq[u]
            valid_ds[u] = []
            test_ds[u] = []
        else:
            train_ds[u] = seq[u][:-2]
            valid_ds[u] = [seq[u][-2]]
            test_ds[u] = [seq[u][-1]]

    return train_ds, valid_ds, test_ds


def df2mat(df, vocab, implicit=True):
    """Transform pandas DataFrame to scipy csr_matrix.

    # coo_matrix object is not subscriptable
    https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.coo_matrix.html

    mat = dok_matrix((n_user, n_item), dtype=np.float32)
    for i in range(len(df)):
        uid, iid, label = df.iloc[i][['user_id', 'item_id', 'label']]
        label = 1 if implicit else label
        mat[uid, iid] = label
        if (i + 1) % 10000 == 0:
            print("processed %d user-item pairs." % (i + 1))
    mat = mat.tocsr()
    """

    assert 'label' in df.columns
    df = df.groupby(['user_id', 'item_id'])['label'].count().reset_index()

    n_user, n_item = vocab['n_user'], vocab['n_item']
    if n_user is None or n_item is None:
        n_user = len(df['user_id'].unique())
        n_item = len(df['item_id'].unique())

    row = df['user_id'].tolist()
    col = df['item_id'].tolist()
    if implicit:
        # data = np.ones(len(df))
        data = df['label'].apply(lambda v: 1 if v >= 1 else 0).tolist()
    else:
        data = df['label'].tolist()
    mat = sp.coo_matrix((data, (row, col)), shape=(n_user, n_item))
    mat = mat.tocsr()
    assert (n_user == mat.shape[0]) & (n_item == mat.shape[1])
    print(mat.shape, mat.nnz)

    return mat


def mat2df(mat, n_user=None, n_item=None, implicit=True):
    """Transform scipy csr_matrix to pandas DataFrame.
    https://docs.scipy.org/doc/numpy/reference/generated/numpy.nonzero.html
    https://docs.scipy.org/doc/numpy/reference/generated/numpy.matrix.html
    """
    uid, iid = mat.nonzero()
    values = mat[np.nonzero(mat)]
    if implicit:
        values = (values > 0).astype(np.int)
    values = values.tolist()[0]
    df = pd.DataFrame(data={'user_id': list(uid), 'item_id': list(iid), 'label': values})
    return df


def split_dataset(mat, seed=1):
    """stratified sampling.
    Split data into train and test subsets in Leave-One-Out manner.
    https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.csr_matrix.html
    https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.csr_matrix.eliminate_zeros.html
    """
    np.random.seed(seed=seed)
    n_user, n_item = mat.shape
    train_mat = mat.copy()
    test_mat = csr_matrix((n_user, n_item), dtype=np.int8)
    skip_cnt = 0
    for u in range(n_user):
        _, items = mat[u].nonzero()
        if len(items) <= 1:
            skip_cnt += 1
            # print("split_dataset, skip entry: ", u, items)
            continue
        idx = np.random.randint(len(items))
        train_mat[u, items[idx]] = 0
        test_mat[u, items[idx]] = 1
    train_mat.eliminate_zeros()
    print("split_dataset, # of total users: {}, # of skip users: {}".format(n_user, skip_cnt))
    return train_mat, test_mat


def sample_dataset(mat, neg_ratio=1):
    if not sp.isspmatrix_dok(mat):
        mat = mat.todok()
    users, items, label = [], [], []
    n_user, n_item = mat.shape
    for (u, v) in mat.keys():
        # positive sample
        users.append(u)
        items.append(v)
        label.append(1)
        # negative samples
        for t in range(neg_ratio):
            i = np.random.randint(n_item)
            while (u, i) in mat:
                i = np.random.randint(n_item)
            users.append(u)
            items.append(i)
            label.append(0)
    users = np.array(users).reshape(-1, 1)
    items = np.array(items).reshape(-1, 1)
    label = np.array(label).reshape(-1, 1)
    return users, items, label


def sample_dataset_df(df, neg_ratio=1):
    n_pos = df['label'].sum()
    # n_pos = (df['label'] == 1).sum()
    df_pos = df[df['label'] == 1]
    df_neg = df[df['label'] == 0]
    # df.sample(frac=0.5, replace=True, random_state=1)
    df_neg = df_neg.sample(n=n_pos * neg_ratio, replace=True, random_state=1)
    df = pd.concat([df_pos, df_neg])
    print("label distribution: \n", df['label'].value_counts())
    print("# of positive samples: ", df['label'].sum())
    print("# of negative samples: ", len(df) - df['label'].sum())
    users = df['role_id'].values
    items = df['bundle_id'].values
    label = df['label'].values
    users = np.array(users).reshape(-1, 1)
    items = np.array(items).reshape(-1, 1)
    label = np.array(label).reshape(-1, 1)
    return users, items, label


def sample_dataset_df_(df, neg_ratio=1):
    user_id, item_id, label = [], [], []
    n_item = len(df['item_id'].unique().tolist())
    for i in range(len(df)):
        uid, iid = df.iloc[i][['user_id', 'item_id']]
        # positive sample
        user_id.append(uid)
        item_id.append(iid)
        label.append(1)
        # negative samples
        for t in range(neg_ratio):
            iid = np.random.randint(n_item)
            df_temp = df[(df['user_id'] == uid) & (df['item_id'] == iid)]
            while len(df_temp) > 0:
                iid = np.random.randint(n_item)
                df_temp = df[(df['user_id'] == uid) & (df['item_id'] == iid)]
            user_id.append(uid)
            item_id.append(iid)
            label.append(0)
    # return pd.DataFrame(data={'user_id': user_id, 'item_id': item_id, 'label': label})
    return user_id, item_id, label


def sample_dataset_weight(mat, weight_mat, neg_ratio=1):
    if not sp.isspmatrix_dok(mat):
        mat = mat.todok()
    assert mat.shape == weight_mat.shape
    users, items, label, weight = [], [], [], []
    n_user, n_item = mat.shape
    for (u, v) in mat.keys():
        # positive sample
        users.append(u)
        items.append(v)
        label.append(1)
        w = weight_mat[u, v]
        assert w > 0.0
        weight.append(w)
        # negative samples
        for t in range(neg_ratio):
            i = np.random.randint(n_item)
            while (u, i) in mat:
                i = np.random.randint(n_item)
            users.append(u)
            items.append(i)
            label.append(0)
            weight.append(w)
    users = np.array(users).reshape(-1, 1)
    items = np.array(items).reshape(-1, 1)
    label = np.array(label).reshape(-1, 1)
    weight = np.array(weight).reshape(-1, 1)
    return users, items, label, weight


def sample_dataset_weight_(mat, weight_mat, neg_ratio=1):
    if not sp.isspmatrix_dok(mat):
        mat = mat.todok()
    n = weight_mat.nnz
    p = weight_mat[weight_mat.nonzero()].A.flatten()
    p = p / p.sum()
    indices = np.random.choice(n, n, replace=True, p=p)
    u, v = weight_mat.nonzero()
    users, items, label = [], [], []
    n_user, n_item = mat.shape
    for idx in indices:
        # positive sample
        users.append(u[idx])
        items.append(v[idx])
        label.append(1)
        # negative samples
        for t in range(neg_ratio):
            i = np.random.randint(n_item)
            while (u, i) in mat:
                i = np.random.randint(n_item)
            users.append(u[idx])
            items.append(i[idx])
            label.append(0)

    assert mat.shape == weight_mat.shape
    mean_weight = np.mean(weight_mat)
    # print(type(weight_mat[users, items]), weight_mat[users, items])
    weight = weight_mat[users, items].A.flatten()
    weight[weight <= 0.0] = mean_weight

    users = np.array(users).reshape(-1, 1)
    items = np.array(items).reshape(-1, 1)
    label = np.array(label).reshape(-1, 1)
    weight = np.array(weight).reshape(-1, 1)
    return users, items, label, weight


def sample_triplet(mat, n_user, n_item, batch_size=None):
    if not sp.isspmatrix_dok(mat):
        mat = mat.todok()
    users, items = mat.nonzero()
    if batch_size is None:
        idx = list(range(len(users)))
        random.shuffle(idx)
        # idx = random.sample(list(range(len(users))), len(users))
        batch_size = len(users)
    else:
        idx = np.random.randint(0, high=len(users), size=batch_size)
    uid = users[idx]
    pos = items[idx]
    neg = np.random.choice(range(n_item), batch_size, replace=True)
    for i in range(batch_size):
        while (uid[i], neg[i]) in mat:
            neg[i] = np.random.randint(n_item)
    # TypeError: can't convert np.ndarray of type numpy.int32.
    return uid.astype(int), pos.astype(int), neg.astype(int)


def sample_batch(mat, n_user, n_item, batch_size=1024):
    """每个正样本随机采样一个负样本"""
    if not sp.isspmatrix_dok(mat):
        mat = mat.todok()
    users, items = mat.nonzero()
    idx = np.random.randint(0, high=len(users), size=batch_size)
    uid = users[idx]
    pos = items[idx]
    neg = np.random.choice(range(n_item), batch_size, replace=True)
    # print(idx, uid, pos, neg)
    for i in range(batch_size):
        while (uid[i], neg[i]) in mat:
            neg[i] = np.random.randint(n_item)
    return uid.astype(int), pos.astype(int), neg.astype(int)


def sample_batch_(mat, n_user, n_item, batch_size=1024):
    """每个正样本随机采样一个负样本"""
    users, items = mat.nonzero()
    idx = np.random.randint(0, high=len(users), size=batch_size)
    uid = users[idx]
    pos = items[idx]
    neg = np.random.choice(range(n_item), batch_size, replace=True)
    # print(idx, uid, pos, neg)
    for i in range(batch_size):
        item_list = mat[uid[i]].nonzero()[1]
        while neg[i] in item_list:
            neg[i] = np.random.choice(range(n_item))
    return uid, pos, neg


def sample_negative(mat, test_mat, neg_ratio=99):
    if not sp.isspmatrix_dok(mat):
        mat = mat.todok()
    neg_items = []
    n_user, n_item = mat.shape
    users, bundles = test_mat.nonzero()
    for u in users:
        # negative samples
        tmp = []
        for t in range(neg_ratio):
            i = np.random.randint(n_item)
            while (u, i) in mat:
                i = np.random.randint(n_item)
            tmp.append(i)
        neg_items.append(tmp)
    return np.array(neg_items)


def sample_negative_df(df, neg_ratio=99):
    neg_sample = []
    item_set = set(df['item_id'].unique().tolist())
    grouped = df.groupby('user_id')
    for _, group in grouped:
        pos_items = group['item_id'].unique().tolist()
        pos_items = set(pos_items)
        z = item_set.difference(pos_items)
        z = list(z)
        if len(z) >= neg_ratio:
            neg_item_id = random.sample(z, neg_ratio)
        else:
            # neg_item_id = z
            neg_item_id = [random.choice(z) for _ in range(neg_ratio)]
        uid = group['user_id'].unique()[0]
        neg_sample.append([uid, neg_item_id])
    return pd.DataFrame(neg_sample, columns=['user_id', 'neg_item_id'])


def chunks(arr, k):
    """arr是被分割的list，每个chunk中含k元素
    list(range(0, 100, 10))
    """
    return [arr[i:i + k] for i in range(0, len(arr), k)]


def negative_sampling_fast(df, neg_ratio=5):
    """Negative Sampling.
    为数据集中的每个正样本采样若干个样本作为负样本
    训练时，用于提供标记信息；测试时，用于评价模型性能
    https://docs.python.org/3/library/random.html#random.sample
    """
    item_set = set(df['item_id'].unique().tolist())

    def func(group, neg_ratio):
        pos_items = group['item_id'].unique().tolist()
        pos_items = set(pos_items)
        z = item_set.difference(pos_items)
        z = list(z)
        n_pos = len(group)
        # print(len(z), n_pos, neg_ratio)
        if len(z) >= neg_ratio * n_pos:
            temp = random.sample(z, neg_ratio * n_pos)
        else:
            # temp = z
            temp = [random.choice(z) for _ in range(neg_ratio * n_pos)]
        neg_item_id = chunks(temp, neg_ratio)
        assert n_pos == len(neg_item_id)
        group['neg_item_id'] = neg_item_id
        return group

    df = df.groupby('user_id').apply(func, neg_ratio)
    return df


def negative_sampling_slow(df, neg_ratio=5):
    """Negative Sampling.
    为数据集中的每个正样本采样若干个样本作为负样本
    训练时，用于提供标记信息；测试时，用于评价模型性能
    https://docs.python.org/3/library/random.html#random.sample
    """
    item_set = set(df['item_id'].unique().tolist())
    group_list = []
    grouped = df.groupby('user_id')
    for _, group in grouped:
        pos_items = group['item_id'].unique().tolist()
        pos_items = set(pos_items)
        z = item_set.difference(pos_items)
        z = list(z)
        n_pos = len(group)
        # print(len(z), n_pos, neg_ratio)
        if len(z) >= neg_ratio * n_pos:
            temp = random.sample(z, neg_ratio * n_pos)
        else:
            # temp = z
            temp = [random.choice(z) for _ in range(neg_ratio * n_pos)]
        neg_item_id = chunks(temp, neg_ratio)
        assert n_pos == len(neg_item_id)
        group['neg_item_id'] = neg_item_id
        group_list.append(group)
    df = pd.concat(group_list)
    # df = df.reset_index(drop=True)
    return df


def groupwhile(pred, seq):
    """Generate lists of elements taken from seq.  Each list will contain
       at least one item, and will also include subsequent items as long as
       pred(group) evaluates to True for the proposed group.

    gen = groupwhile(lambda group: len(group) <= 5 and sum(group) <= 50, itertools.count(1))
    for batch in gen:
        print('sum {} = {}'.format(batch, sum(batch)))
    """
    seq = iter(seq)
    try:
        group = [next(seq)]
    except StopIteration:
        pass
    else:
        for item in seq:
            if pred(group + [item]):
                group.append(item)
            else:
                yield group
                group = [item]
        yield group


def batch_generator(data, batch_size=1024, shuffle=True, drop_last=True):
    """
    https://docs.python.org/3/library/itertools.html
    https://stackoverflow.com/questions/8290397/how-to-split-an-iterable-in-constant-size-chunks
    https://stackoverflow.com/questions/55889923/how-to-handle-the-last-batch-using-keras-fit-generator
    https://stanford.edu/~shervine/blog/keras-how-to-generate-data-on-the-fly
    https://pytorch.org/docs/stable/data.html
    https://pytorch.org/docs/stable/_modules/torch/utils/data/dataloader.html#DataLoader
    # number of batches per epoch
    samples = len(x)
    batches = int(np.floor(samples / batch_size))
    batches = int(np.ceil(samples / batch_size))
    data = list(data)
    for i in range(len(data)):
        if isinstance(data[i], (list, tuple)):
            data[i] = np.array(data[i])
    """
    indices = np.arange(len(data[0]))
    if shuffle:
        np.random.shuffle(indices)
        # indexes = np.random.permutation(len(x))
    idx = 0
    while True:
        start = idx * batch_size
        if start + batch_size > len(data[0]):
            if drop_last:
                pass
            idx = 0
            start = 0 * batch_size
            if shuffle:
                np.random.shuffle(indices)
        batch_indices = indices[start:start + batch_size]
        # batch_indices = indices[idx * batch_size:(idx + 1) * batch_size]
        # print(batch_indices)
        idx += 1
        yield [d[batch_indices] for d in data]


def batch_generator_one_shot(data, batch_size=1024, shuffle=True, drop_last=True):
    """batch generator with one shot iterator
    https://www.tensorflow.org/api_docs/python/tf/compat/v1/data/make_one_shot_iterator
    """
    samples = len(data[0])
    indices = np.arange(samples)
    if shuffle:
        np.random.shuffle(indices)
    # batches = int(np.floor(samples / batch_size))
    batches = int(np.ceil(samples / batch_size))
    for idx in range(batches):
        start = idx * batch_size
        if idx == batches - 1:
            if drop_last:
                return None
            else:
                batch_indices = indices[start:]
        else:
            batch_indices = indices[start:start + batch_size]
        yield [d[batch_indices] for d in data]


def batch_generator_x_y(x, y=None, batch_size=1024, shuffle=True, drop_last=True):
    """batch generator with features and labels.
    """
    if y is not None:
        assert len(x) == len(y)
    indices = np.arange(len(x))
    if shuffle:
        np.random.shuffle(indices)
    idx = 0
    while True:
        start = idx * batch_size
        if start + batch_size > len(x):
            if drop_last:
                pass
            idx = 0
            start = 0 * batch_size
            if shuffle:
                np.random.shuffle(indices)
        # batch_indices = indices[start:start + batch_size]
        # batch_indices = indices[idx * batch_size:(idx + 1) * batch_size]
        idx += 1
        batch_x = x[indices[start:start + batch_size]]
        batch_y = y[indices[start:start + batch_size]] if y is not None else None
        yield batch_x, batch_y
