"""
@Author  : vwonx
@Date    : 2020/4/28
"""

from collections import OrderedDict

from lib.utils import *

import torch
from torch import nn


class DeepLabLargeFOV(nn.Module):
    def __init__(self, args):
        super(DeepLabLargeFOV, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, 3, 1, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, 3, 1, 1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(3, 2, 1, ceil_mode=True),

            nn.Conv2d(64, 128, 3, 1, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, 3, 1, 1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(3, 2, 1, ceil_mode=True),

            nn.Conv2d(128, 256, 3, 1, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, 3, 1, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, 3, 1, 1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(3, 2, 1, ceil_mode=True),

            nn.Conv2d(256, 512, 3, 1, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, 3, 1, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, 3, 1, 1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(3, 1, 1, ceil_mode=True),

            nn.Conv2d(512, 512, 3, 1, 2, 2),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, 3, 1, 2, 2),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, 3, 1, 2, 2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(3, 1, 1, ceil_mode=True),
            nn.AvgPool2d(3, 1, 1, ceil_mode=True),
        )

        self.fc6 = nn.Conv2d(512, 1024, 3, 1, 12, 12)
        self.relu6 = nn.ReLU(inplace=True)
        self.dropout6 = nn.Dropout2d(0.5)

        self.fc7 = nn.Conv2d(1024, 1024, 1)
        self.relu7 = nn.ReLU()
        self.dropout7 = nn.Dropout2d(0.5)

        self.fc8 = nn.Conv2d(1024, args.num_cls, 1)
        nn.init.normal_(self.fc8.weight, mean=0, std=0.01)
        nn.init.constant_(self.fc8.bias, 0)

        self.infer = args.infer

        info(None, "Using pretrained model: {}".format(args.pretrained), 'red')
        self.load_weight(args.pretrained)

    def forward(self, x):
        x = self.features(x)
        x = self.fc6(x)
        x = self.relu6(x)
        x = self.dropout6(x)
        x = self.fc7(x)
        x = self.relu7(x)
        x = self.dropout7(x)
        x = self.fc8(x)

        return x

    def load_weight(self, pre_train):

        if not self.infer:
            pre_train_model = torch.load(pre_train, map_location=torch.device('cpu'))
            new_state_dict = OrderedDict()

            for k, v in pre_train_model.items():
                if int(k.split('.')[0]) <= 28:
                    new_state_dict['features.' + k] = v
                elif int(k.split('.')[0]) == 31:
                    new_state_dict['fc6.' + k.split('.')[1]] = v
                elif int(k.split('.')[0]) == 34:
                    new_state_dict['fc7.' + k.split('.')[1]] = v

            self.load_state_dict(new_state_dict, strict=False)

        else:
            pre_train_model = torch.load(pre_train, map_location=torch.device('cpu'))
            self.load_state_dict(pre_train_model['net'])
