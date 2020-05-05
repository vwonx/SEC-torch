"""
@Author  : vwonx
@Date    : 2020/4/28
"""

import os
from subprocess import call

from torch import nn
import torch
import cv2


def get_parameters(network, category_num, bias=False, final=False):
    for m in network.modules():
        if isinstance(m, nn.Conv2d):
            if (final and m.out_channels != category_num) or (
                    not final and m.out_channels == category_num):
                continue
            if bias:
                yield m.bias
            else:
                yield m.weight


class SaveModel(object):
    def __init__(self, model, snapshot, model_name, num_save=5):
        super(SaveModel, self).__init__()
        self.model = model
        self.snapshot = snapshot
        self.model_name = model_name
        self.num_save = num_save
        self.save_list = []

    def save(self, n_epoch):
        self.save_list += [os.path.join(self.snapshot, '{}-{:04d}.pth'.format(self.model_name, n_epoch))]
        state = {'net': self.model.network.state_dict(), 'optimizer': self.model.optimizer.state_dict(),
                 'epoch': n_epoch}
        torch.save(state, self.save_list[-1])

        if len(self.save_list) > self.num_save:
            call(['rm', self.save_list[0]])
            self.save_list = self.save_list[1:]
        return self.save_list[-1]

    def __call__(self, n_epoch):
        return self.save(n_epoch)


def imwrite(filename, image):
    dirname = os.path.dirname(filename)
    if not os.path.exists(dirname):
        try:
            os.makedirs(dirname)
        except:
            raise ValueError('Can not create folder to store image.')
    cv2.imwrite(filename, image)
