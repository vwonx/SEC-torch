"""
@Author  : vwonx
@Date    : 2020/4/28
"""

from abc import ABC, abstractmethod

from .network import *
from .loss import *


class BaseModel(ABC):
    def __init__(self, args):
        super(BaseModel, self).__init__()
        self.network = eval(args.backbone)(args).cuda()
        self.loss = eval(args.model + 'Loss')()
        self.optimizer = None
        self.scheduler = None

    @abstractmethod
    def forward_backward(self, batch):
        pass

    @abstractmethod
    def infer(self, images):
        pass

    @abstractmethod
    def update(self):
        pass

    def train(self):
        self.network.train()

    def eval(self):
        self.network.eval()
