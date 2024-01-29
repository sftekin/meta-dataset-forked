# Adapted from https://github.com/wyharveychen/CloserLookFewShot

import torch
import torch.nn as nn
from torch.autograd import Variable
import numpy as np
from abc import abstractmethod


class MetaTemplate(nn.Module):
    def __init__(self, model_func, n_way, n_support, n_query, change_way=True):
        super(MetaTemplate, self).__init__()
        self.n_way = n_way
        self.n_support = n_support
        self.n_query = n_query
        self.feature = model_func()
        self.change_way = change_way  # some methods allow different_way classification during training and test

    @abstractmethod
    def set_forward(self, x, is_feature):
        pass

    @abstractmethod
    def variable_forward(self, episode):
        pass

    @abstractmethod
    def set_forward_loss(self, x, y):
        pass

    def forward(self, x):
        out = self.feature.forward(x)
        return out

    def parse_feature(self, x, is_feature):
        x = Variable(x.cuda())
        if is_feature:
            z_all = x
        else:
            x = x.contiguous().view(self.n_way * (self.n_support + self.n_query), *x.size()[2:])
            z_all = self.feature.forward(x)
            z_all = z_all.view(self.n_way, self.n_support + self.n_query, -1)
        z_support = z_all[:, :self.n_support]
        z_query = z_all[:, self.n_support:]

        return z_support, z_query

    def correct(self, x, y=None, variable=False):
        if y is None:
            y = np.repeat(range(self.n_way), self.n_query)
        else:
            y = y.numpy()
        if variable:
            scores = self.variable_forward(x)
        else:
            scores = self.set_forward(x)

        topk_scores, topk_labels = scores.data.topk(1, 1, True, True)
        topk_ind = topk_labels.cpu().numpy()
        top1_correct = np.sum(topk_ind[:, 0] == y)
        return float(top1_correct), len(y)
