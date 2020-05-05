"""
@Author  : vwonx
@Date    : 2020/4/27
"""

import pickle
import os

import torch.utils.data as data

import cv2
import numpy as np
import torch
from torch.utils.data import DataLoader


def voc2012_loader(args):
    dataset = VOC2012(args)
    data_loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=not args.infer,
        num_workers=args.num_workers,
    )

    return data_loader


class VOC2012(data.Dataset):
    def __init__(self, args):
        super(VOC2012, self).__init__()

        self.image_root = args.image_root
        self.label_root = args.label_root
        self.data_list = args.data_list

        self.mean_value = [104.0, 117.0, 123.0]
        self.image_size = args.image_size

        self.infer = args.infer
        self.num_cls = args.num_cls

        with open(self.data_list, 'r') as f:
            if args.infer:
                self.image_names = [image_name for image_name in f.read().splitlines()]
            else:
                self.image_names = [image_name.split()[0][:-4] for image_name in f.read().splitlines()]

        if not self.infer:
            with open(self.label_root, 'rb') as seed:
                self.seed = pickle.load(seed)

    def __len__(self):
        return len(self.image_names)

    def __getitem__(self, index):
        name = self.image_names[index]
        image_path = os.path.join(self.image_root, name + '.jpg')

        origin_image = cv2.imread(image_path, cv2.IMREAD_COLOR)
        image = self._preprocess(origin_image)
        image = torch.from_numpy(image)

        if self.infer:
            return name, origin_image, image
        else:
            label, seed = self._get_seed(index)

            label = torch.from_numpy(label)
            seed = torch.from_numpy(seed)
            return image, label, seed

    def _preprocess(self, image):
        image = cv2.resize(image, (self.image_size, self.image_size))
        image = np.array(image)

        image = image - np.array(self.mean_value)
        image = image.transpose(2, 0, 1)
        return image.astype(np.float32)

    def _get_seed(self, index):
        label = np.zeros((1, 1, self.num_cls))
        label[0, 0, self.seed['%s_labels' % index]] = 1

        seed = np.zeros((self.num_cls, 41, 41))
        seed_index = self.seed['%s_cues' % index]
        seed[seed_index[0], seed_index[1], seed_index[2]] = 1

        return label.astype(np.float32), seed.astype(np.float32)
