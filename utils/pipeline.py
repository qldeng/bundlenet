"""
Base Classes and Functions (Regression and Classification).
https://keras.io/api/models/
"""

import time

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from utils.data_utils import batch_generator, batch_generator_one_shot

device = torch.device("cpu")
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class Pipeline(object):

    def __init__(self, model, config):
        super(Pipeline, self).__init__()
        self.model = model
        self.optimizer = None
        self.loss_fn = None
        self.metrics = None

        # configuration
        self.verbose = config.get('verbose', False)
        self.model_dir = config.get('model_dir', '.')
        self.device = config['device']
        # self.device = torch.device(config['device'])
        # if config['device'] == 'cpu':
        #     device = torch.device('cpu')
        # else:
        #     device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        # self.device = device

        if self.verbose:
            print('-' * 80)
            for idx, m in enumerate(model.modules()):
                print(idx, '->', m)
            print('-' * 80)
            for param in model.parameters():
                print(type(param.data), param.size())
            print('-' * 80)
            print(model.state_dict().keys())
            print('-' * 80)

    def compile(self, optimizer, loss_fn, metrics=None):
        """
        https://keras.io/models/model/#compile
        https://keras.io/initializers/
        https://pytorch.org/docs/stable/nn.init
        model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
        :param optimizer:
        :param loss_fn:
        :param metrics:
        :return:
        """
        self.optimizer = optimizer
        self.loss_fn = loss_fn
        self.metrics = metrics

    def _optimizer(self, optim='adam', lr=None):
        """optimization algorithms.
        https://pytorch.org/docs/stable/optim.html
        """
        # "self.parameters()" is a generator object
        # print(type(self.parameters()), self.parameters())
        params = self.model.parameters()
        # params = list(self.parameters())
        if optim == 'sgd':
            optimizer = torch.optim.SGD(params, lr=lr, momentum=0, weight_decay=0)
        elif optim == 'adam':
            # optimizer = torch.optim.Adam(params, lr=0.001, betas=(0.9, 0.999), eps=1e-08, weight_decay=0)
            optimizer = torch.optim.Adam(params, lr=0.01, weight_decay=5e-4)
        elif optim == 'adagrad':
            optimizer = torch.optim.Adagrad(params, lr=0.01, lr_decay=0, weight_decay=0)
        elif optim == 'rmsprop':
            optimizer = torch.optim.RMSprop(params, lr=0.01, alpha=0.99, eps=1e-08, weight_decay=0)
        else:
            optimizer = torch.optim.Adam(params, lr=lr, betas=(0.9, 0.999), eps=1e-08, weight_decay=0)
        return optimizer

    def fit(self, inputs, batch_size=None, n_epochs=1):
        """model fit.
        inputs: (x_u, x_i, y), both input and output are numpy arrays.
        https://keras.io/models/model/#fit
        """
        history = []
        for epoch in range(n_epochs):
            start_time = time.time()
            self.model.train()
            if batch_size is None:
                train_loss, train_acc = self.train_batch(inputs)
                history.append((train_loss, train_acc))
            else:
                train_loss, train_acc = self.train(inputs, batch_size)
                history.append((train_loss, train_acc))
            # self.model.eval()
            # if batch_size is None:
            #     loss, metrics = self.evaluate_batch(inputs)
            #     history.append((loss, metrics))
            # else:
            #     loss, metrics = self.evaluate(inputs, batch_size)
            #     history.append((loss, metrics))
            secs = int(time.time() - start_time)
            mins = secs // 60
            secs = secs % 60
            # print('Epoch: %d' % (epoch + 1), " | time in %d minutes, %d seconds" % (mins, secs))
            # ValueError: Unknown format code 'd' for object of type 'float'
            # print(f'\tEpoch: {epoch + 1} | time in {mins:d} minutes {secs} seconds')
            # print(f'\tLoss: {train_loss:.4f}(train)\t|\tAcc: {train_acc * 100:.1f}%(train)')
            # print(f'\tLoss: {valid_loss:.4f}(valid)\t|\tAcc: {valid_acc * 100:.1f}%(valid)')
        return history

    def fit_triplet(self, inputs, batch_size=None, n_epochs=1):
        x_u, x_p, x_n = inputs
        x_u = torch.tensor(x_u, dtype=torch.long).to(self.device).view(-1, 1)
        x_p = torch.tensor(x_p, dtype=torch.long).to(self.device).view(-1, 1)
        x_n = torch.tensor(x_n, dtype=torch.long).to(self.device).view(-1, 1)
        output_pos = self.model((x_u, x_p))
        output_neg = self.model((x_u, x_n))
        y = torch.ones(len(x_u)).float()
        # print(output_pos.shape, output_neg.shape, y.shape)
        # https://pytorch.org/docs/master/nn.functional.html#margin-ranking-loss
        loss = F.margin_ranking_loss(output_pos, output_neg, y, margin=0.1)
        # loss = -torch.sum(torch.log(torch.sigmoid(output_pos - output_neg)))
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return loss.item(), None

    def _metrics(self, y_true, y_pred):
        """calculate model metrics.
        https://keras.io/metrics/
        https://scikit-learn.org/stable/modules/classes.html#module-sklearn.metrics
        """
        a = y_pred.mean().item()
        y_pred = torch.round(torch.sigmoid(y_pred))
        # print(y_true.sum().item(), "/", len(y_true), y_pred.sum().item(), "/", len(y_pred), a)
        correct = torch.eq(y_pred.type(y_true.type()), y_true).view(-1)
        accuracy = correct.sum().item() / correct.shape[0]
        return accuracy

    def predict(self, inputs, batch_size):
        """model predict
        output = model(input). both inputs and outputs are numpy arrays.
        https://keras.io/models/model/#predict
        """
        self.model.eval()
        batch_gen = batch_generator_one_shot(inputs, batch_size=batch_size, shuffle=False, drop_last=False)
        result = []
        for batch_inputs in batch_gen:
            pred = self.predict_batch(batch_inputs)
            result.append(pred)
        result = np.concatenate(result, axis=0)
        return np.array(result)

    def predict_batch(self, inputs):
        """evaluate on batch.
        https://keras.io/models/model/#test_on_batch
        """
        x_u, x_i = inputs
        x_u = torch.tensor(x_u, dtype=torch.long).to(self.device).view(-1, 1)
        x_i = torch.tensor(x_i, dtype=torch.long).to(self.device).view(-1, 1)
        with torch.no_grad():
            pred = self.model(x_u, x_i)
            # pred = torch.round(torch.sigmoid(output))
            # pred = pred.detach().numpy() if self.device == 'cpu' else pred.cpu().detach().numpy()
        return pred

    def train(self, inputs, batch_size):
        self.model.train()
        steps_per_epoch = int(np.floor(len(inputs[0]) / batch_size))
        log = "number of samples: {}, batch size: {}, steps per epoch: {}"
        print(log.format(len(inputs[0]), batch_size, steps_per_epoch))
        batch_gen = batch_generator_one_shot(inputs, batch_size=batch_size, shuffle=True, drop_last=False)
        history = []
        for i in range(steps_per_epoch):
            batch_inputs = next(batch_gen)
            loss, metrics = self.train_batch(batch_inputs)
            history.append((loss, metrics))
        loss, metrics = np.mean(np.array(history), axis=0)
        return loss, metrics

    def train_batch(self, inputs):
        """fit on batch.
        https://keras.io/models/model/#train_on_batch
        """
        x_u, x_i, y = inputs
        assert len(x_u) == len(x_i)
        x_u = torch.tensor(x_u, dtype=torch.long).to(self.device).view(-1, 1)
        x_i = torch.tensor(x_i, dtype=torch.long).to(self.device).view(-1, 1)
        y = torch.tensor(y, dtype=torch.float).to(self.device)
        # y = torch.tensor(label).float().to(self.device)
        output = self.model(x_u, x_i)  # (batch_size, 1)
        # loss = F.binary_cross_entropy_with_logits(output, y)
        loss = self.loss_fn(output, y)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        metrics = self._metrics(y, output)
        return loss.item(), metrics

    def train_triplet(self, inputs, epochs=1, batch_size=None, shuffle=True):
        pass

    def train_triplet_batch(self, input):
        users, pos_items, neg_items = input
        self.optimizer.zero_grad()
        output_pos = self.model(users, pos_items)
        output_pos = output_pos.view(-1)
        output_neg = self.model(users, neg_items)
        output_neg = output_neg.view(-1)
        target = torch.ones(len(users)).float().view(-1)
        # print(output_pos, output_pos.shape)
        # print(output_neg, output_neg.shape)
        # print(target, target.shape)
        loss = F.margin_ranking_loss(output_pos, output_neg, target, margin=1.0)
        # loss = -torch.mean(torch.log(torch.sigmoid(output_pos - output_neg)))
        loss.backward()
        self.optimizer.step()
        return loss

    def evaluate(self, inputs, batch_size):
        """model evaluate.
        https://keras.io/models/model/#evaluate
        """
        self.model.eval()
        batch_gen = batch_generator_one_shot(inputs, batch_size=batch_size, shuffle=False, drop_last=False)
        history = []
        for batch_inputs in batch_gen:
            loss, metrics = self.evaluate_batch(batch_inputs)
            history.append((loss, metrics))
        loss, metrics = np.mean(np.array(history), axis=0)
        return loss, metrics

    def evaluate_batch(self, inputs):
        """evaluate on batch.
        https://keras.io/models/model/#test_on_batch
        """
        x_u, x_i, y = inputs
        x_u = torch.tensor(x_u, dtype=torch.long).to(self.device)
        x_i = torch.tensor(x_i, dtype=torch.long).to(self.device)
        y = torch.tensor(y, dtype=torch.float).to(self.device).view(-1, 1)
        with torch.no_grad():
            output = self.model(x_u, x_i)
            # loss = F.binary_cross_entropy_with_logits(output, y)
            loss = self.loss_fn(output, y)
            metrics = self._metrics(y, output)
        return loss.item(), metrics
