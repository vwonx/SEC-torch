"""
@Author  : vwonx
@Date    : 2020/4/28
"""

from .base_model import BaseModel
from lib.utils import get_parameters

import torch
from torch.optim import lr_scheduler


class SEC(BaseModel):
    def __init__(self, args):
        super(SEC, self).__init__(args)
        self.optimizer = torch.optim.SGD([
            {'params': get_parameters(self.network, args.num_cls, bias=False, final=False),
             'lr': args.lr, 'weight_decay': args.weight_decay},
            {'params': get_parameters(self.network, args.num_cls, bias=True, final=False),
             'lr': args.lr * 2, 'weight_decay': 0},
            {'params': get_parameters(self.network, args.num_cls, bias=False, final=True),
             'lr': args.lr * 10, 'weight_decay': args.weight_decay},
            {'params': get_parameters(self.network, args.num_cls, bias=True, final=True),
             'lr': args.lr * 20, 'weight_decay': 0}
        ], momentum=args.momentum)
        self.scheduler = lr_scheduler.StepLR(self.optimizer, step_size=args.step, gamma=args.gamma)

        self.output = None

    def forward_backward(self, batch):
        # Input
        images, labels, seed = batch
        images = images.cuda()
        labels = labels.cuda()
        seed = seed.cuda()

        # Forward
        self.output = self.network(images)
        loss = self.loss(images, self.output, labels, seed)

        # Backward
        self.optimizer.zero_grad()
        loss.backward()

    def infer(self, images):
        images = images.cuda()
        self.output = self.network(images)

    def update(self):
        self.optimizer.step()
        self.scheduler.step()
