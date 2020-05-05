"""
@Author  : vwonx
@Date    : 2020/4/28
"""

# reference: https://github.com/halbielee/SEC_pytorch/blob/master/utils/util_loss.py

from torch import nn
import torch
import numpy as np
from scipy.ndimage import zoom
from krahenbuhl2013 import CRF


class SECLoss(nn.Module):
    def __init__(self):
        super(SECLoss, self).__init__()
        self.min_prob = 1e-4
        self.losses = {}

    def forward(self, images, output, label, gt_cue):
        fc8_sec_softmax = self.softmax_layer(output)
        loss_s = self.seed_loss_layer(fc8_sec_softmax, gt_cue)
        loss_e = self.expand_loss_layer(fc8_sec_softmax, label)
        fc8_sec_crf_log = self.crf_layer(output, images)
        loss_c = self.constrain_loss_layer(fc8_sec_softmax, fc8_sec_crf_log)

        self.losses['loss_s'] = loss_s.item()
        self.losses['loss_e'] = loss_e.item()
        self.losses['loss_c'] = loss_c.item()

        loss = loss_s + loss_e + loss_c
        self.losses['loss'] = loss.item()
        return loss

    def get_loss(self):
        return self.losses

    def softmax_layer(self, preds):
        preds = preds
        pred_max, _ = torch.max(preds, dim=1, keepdim=True)
        pred_exp = torch.exp(preds - pred_max.clone().detach())
        probs = pred_exp / torch.sum(pred_exp, dim=1, keepdim=True) + self.min_prob
        probs = probs / torch.sum(probs, dim=1, keepdim=True)
        return probs

    def seed_loss_layer(self, probs, labels):
        count = torch.sum(labels, dim=[1, 2, 3], keepdim=True)
        loss_balanced = - torch.mean(torch.sum(labels * torch.log(probs), dim=[1, 2, 3], keepdim=True) / count)
        return loss_balanced

    def expand_loss_layer(self, probs_tmp, stat_inp):
        stat = stat_inp[:, :, :, 1:]

        probs_bg = probs_tmp[:, 0, :, :]
        probs = probs_tmp[:, 1:, :, :]

        probs_max, _ = torch.max(torch.max(probs, dim=3)[0], dim=2)

        q_fg = 0.996
        probs_sort, _ = torch.sort(probs.contiguous().view(-1, 20, 41 * 41), dim=2)
        weights = probs_sort.new_tensor([q_fg ** i for i in range(41 * 41 - 1, -1, -1)])[None, None, :]
        z_fg = torch.sum(weights)
        probs_mean = torch.sum((probs_sort * weights) / z_fg, dim=2)

        q_bg = 0.999
        probs_bg_sort, _ = torch.sort(probs_bg.contiguous().view(-1, 41 * 41), dim=1)
        weights_bg = probs_sort.new_tensor([q_bg ** i for i in range(41 * 41 - 1, -1, -1)])[None, :]
        z_bg = torch.sum(weights_bg)
        # weights_bg = ..
        probs_bg_mean = torch.sum((probs_bg_sort * weights_bg) / z_bg, dim=1)

        stat_2d = (stat[:, 0, 0, :] > 0.5).float()
        loss_1 = -torch.mean(
            torch.sum((stat_2d * torch.log(probs_mean) / torch.sum(stat_2d, dim=1, keepdim=True)), dim=1))
        loss_2 = -torch.mean(
            torch.sum(((1 - stat_2d) * torch.log(1 - probs_max) / torch.sum(1 - stat_2d, dim=1, keepdim=True)), dim=1))
        loss_3 = -torch.mean(torch.log(probs_bg_mean))

        loss = loss_1 + loss_2 + loss_3
        return loss

    def constrain_loss_layer(self, probs, probs_smooth_log):
        probs_smooth = torch.exp(probs.new_tensor(
            probs_smooth_log, requires_grad=True))
        loss = torch.mean(
            torch.sum(probs_smooth * torch.log(probs_smooth / probs), dim=1))

        return loss

    def crf_layer(self, fc8_SEC, images):
        unary = np.transpose(np.array(fc8_SEC.cpu().clone().data), [0, 2, 3, 1])
        mean_pixel = np.array([104.0, 117.0, 123.0])
        im = images.cpu().data
        im = zoom(im, (1, 1, 41 / im.shape[2], 41 / im.shape[3]), order=1)

        im = im + mean_pixel[None, :, None, None]
        im = np.transpose(np.round(im), [0, 2, 3, 1])

        N = unary.shape[0]

        result = np.zeros(unary.shape)

        for i in range(N):
            result[i] = CRF(im[i], unary[i], scale_factor=12.0)
        result = np.transpose(result, [0, 3, 1, 2])
        result[result < self.min_prob] = self.min_prob
        result = result / np.sum(result, axis=1, keepdims=True)

        return np.log(result)
