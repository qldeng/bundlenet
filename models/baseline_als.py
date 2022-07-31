"""
https://implicit.readthedocs.io/en/latest/quickstart.html
https://implicit.readthedocs.io/en/latest/als.html
https://machinelearningmastery.com/sparse-matrices-for-machine-learning/
https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.csr_matrix.html
https://blog.csdn.net/power0405hf/article/details/47789481
"""

import implicit


class BayesianPersonalizedRanking(object):
    """
    二值矩阵分解模型
    """

    def __init__(self, config):
        self.verbose = config.get('verbose', False)

        # data setting
        self.n_user = config['n_user']
        self.n_item = config['n_item']
        # self.input_dim = config['input_dim']  # # of features
        # self.output_dim = config['output_dim'] or 1  # # of class

        self.n_factor = config.get('n_factor', 128)
        self.algo = config.get('model_type', 'bpr')  # model type (als, bpr, lmf)

        self.model = None
        self.data = None

    def fit(self, mat):
        """训练模型，并进行离线的推理预测
        """
        # user-item interaction data
        self.data = mat

        item_user = mat.T.tocsr()

        if self.algo == 'als':
            # initialize a model
            model = implicit.als.AlternatingLeastSquares(factors=self.n_factor,
                                                         regularization=0.01,
                                                         iterations=50,
                                                         calculate_training_loss=True,
                                                         use_gpu=False)
            # train the model on a sparse matrix of item/user/confidence weights
            model.fit(item_user, show_progress=True)

        elif self.algo == 'lmf':
            model = implicit.lmf.LogisticMatrixFactorization(factors=self.n_factor,
                                                             learning_rate=0.01,
                                                             regularization=0.01,
                                                             iterations=50,
                                                             neg_prop=10,
                                                             use_gpu=False)
            item_user = item_user.tocoo()
            model.fit(item_user, show_progress=True)
        elif self.algo == 'bpr':
            model = implicit.bpr.BayesianPersonalizedRanking(factors=self.n_factor,
                                                             learning_rate=0.01,
                                                             regularization=0.01,
                                                             iterations=100,
                                                             verify_negative_samples=True,
                                                             use_gpu=False)
            item_user = item_user.tocoo()
            model.fit(item_user, show_progress=True)
        else:
            model = implicit.als.AlternatingLeastSquares(factors=self.n_factor,
                                                         regularization=0.01,
                                                         iterations=50,
                                                         calculate_training_loss=True,
                                                         use_gpu=True)
            model.fit(item_user, show_progress=True)

        self.model = model

        return model

    def predict(self, input, batch_size=None):
        user_id, item_id = list(input[0]), list(input[1])
        pred = []
        for i in range(len(user_id)):
            uid, iid = user_id[i], item_id[i]
            # https://github.com/benfred/implicit/issues/253
            pred_ = self.model.rank_items(userid=uid, user_items=self.data, selected_items=[iid])
            pred.append(pred_[0][1])
        return pred

    def rank(self, uid, item_list, top_k):
        result = self.model.rank_items(userid=uid, user_items=self.data, selected_items=item_list)
        rank_list = [v[0] for v in result[0:top_k]]
        return rank_list

    def recommend(self, user_id):
        result = []
        for uid in user_id:
            recoms = self.model.recommend(userid=uid,
                                          user_items=self.data,
                                          N=10,
                                          filter_already_liked_items=True,
                                          filter_items=None,
                                          recalculate_user=False)
            result.append(recoms)
        return result


if __name__ == '__main__':

    pass
