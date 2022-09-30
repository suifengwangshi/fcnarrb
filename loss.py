#!/usr/bin/python
# -*- encoding: utf-8 -*-

import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np


class OhemNegLoss(nn.Module):
    def __init__(self, device, thresh=0.3):
        super(OhemNegLoss, self).__init__()
        thresh = torch.tensor(thresh)
        self.thresh = thresh.to(device)
        self.criteria = nn.BCELoss(reduction='none')

    def forward(self, denselabel_p, denselabel_t):
        # hard negative example mining
        denselabel_p_v = denselabel_p.view(-1)
        denselabel_t_v = denselabel_t.view(-1)
        index_pos = (denselabel_t_v == 1)
        index_neg = (denselabel_t_v == 0)
        denselabel_p_pos = denselabel_p_v[index_pos]
        denselabel_t_pos = denselabel_t_v[index_pos]
        denselabel_p_neg = denselabel_p_v[index_neg]
        denselabel_t_neg = denselabel_t_v[index_neg]

        loss_pos = self.criteria(denselabel_p_pos, denselabel_t_pos)
        loss_neg = self.criteria(denselabel_p_neg, denselabel_t_neg)
        loss_neg, _ = torch.sort(loss_neg, descending=True)
        number_neg = int(self.thresh*loss_neg.numel())
        loss_neg = loss_neg[:number_neg]
        loss = torch.mean(loss_pos) + torch.mean(loss_neg)
        return loss


class FacalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2, size_average=True):
        super(FacalLoss, self).__init__()
        self.alpha = torch.tensor(alpha).cuda()
        self.gamma = gamma
        self.size_average = size_average

    def forward(self, denselabel_p, denselabel_t):
        denselabel_p_v = denselabel_p.view(-1, 1)
        denselabel_t_v = denselabel_t.view(-1, 1)
        print(denselabel_p_v.shape)

        denselabel_p_v = torch.cat((1-denselabel_p_v, denselabel_p_v), dim=1)

        class_mask = torch.zeros(denselabel_p_v.shape[0], denselabel_p_v.shape[1]).cuda()
        class_mask.scatter_(1, denselabel_t_v.view(-1, 1).long(), 1.)

        probs = (denselabel_p_v * class_mask).sum(dim=1).view(-1, 1)
        probs = probs.clamp(min=0.0001, max=1.0)

        log_p = probs.log()

        alpha = torch.ones(denselabel_p_v.shape[0], denselabel_p_v.shape[1]).cuda()
        alpha[:, 0] = alpha[:, 0] * (1-self.alpha)
        alpha[:, 1] = alpha[:, 1] * self.alpha
        alpha = (alpha * class_mask).sum(dim=1).view(-1, 1)

        batch_loss = -alpha*(torch.pow((1-probs), self.gamma)) * log_p

        if self.size_average:
            loss = batch_loss.mean()
        else:
            loss = batch_loss.sum()

        return loss



