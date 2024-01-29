# Adapted from https://github.com/wyharveychen/CloserLookFewShot

import os
import methods.backbone
import torch
import torch.nn as nn
from torch.autograd import Variable
import numpy as np
import torch.nn.functional as F
from methods.meta_template import MetaTemplate
from collections import Counter


class ProtoNet(MetaTemplate):
    def __init__(self, model_func, n_way, n_support, n_query):
        super(ProtoNet, self).__init__(model_func, n_way, n_support, n_query)
        self.loss_fn = nn.CrossEntropyLoss()
        self.variable_mean = lambda x: torch.mean(x, dim=0)

    def variable_forward(self, episode):
        im_sup, im_query = episode[0].permute(0, 3, 1, 2).cuda(), episode[3].permute(0, 3, 1, 2).cuda()
        z_support = self.feature.forward(im_sup)
        z_query = self.feature.forward(im_query)
        variable_support = torch.split(z_support, list(Counter(episode[1].numpy()).values()))
        z_proto = torch.stack(list(map(self.variable_mean, variable_support)))
        dists = euclidean_dist(z_query, z_proto)
        scores = -dists
        return scores

    def set_forward(self, x, is_feature=False):
        z_support, z_query = self.parse_feature(x, is_feature)
        z_support = z_support.contiguous()
        z_proto = z_support.view(self.n_way, self.n_support, -1).mean(1)  # the shape of z is [n_data, n_dim]
        z_query = z_query.contiguous().view(self.n_way * self.n_query, -1)

        dists = euclidean_dist(z_query, z_proto)
        scores = -dists
        return scores

    def set_forward_loss(self, x, y=None, variable=False):
        if y is None:
            y = torch.from_numpy(np.repeat(range(self.n_way), self.n_query))
            y = Variable(y.cuda())
        else:
            y = y.type(torch.LongTensor).cuda()
        if variable:
            scores = self.variable_forward(x)
        else:
            scores = self.set_forward(x)

        return self.loss_fn(scores, y)


def euclidean_dist(x, y):
    # x: N x D
    # y: M x D
    n = x.size(0)
    m = y.size(0)
    d = x.size(1)
    assert d == y.size(1)

    x = x.unsqueeze(1).expand(n, m, d)
    y = y.unsqueeze(0).expand(n, m, d)

    return torch.norm(x - y, dim=2, p=2)
