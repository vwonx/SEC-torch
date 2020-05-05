#!/bin/bash
GPU=0

# Config env
export PYTHONPATH=$PYTHONPATH:Path/of/SEC-torch

# Config dataset path
DATASET=/Path/of/voc2012
image_root=$DATASET/JPEGImages
groundtruth_root=$DATASET/extra/SegmentationClassAug

# Config model
backbone=DeepLabLargeFOV
pretrained=data/pretrained/vgg16_20M.pth
seeds=data/seed/localization_cues.pickle
snapshot=./snapshot/SEC/$backbone

#-------------training & testing----------------
#train
python ./scripts/train.py --image-root $image_root --label-root $seeds --snapshot $snapshot --backbone $backbone --pretrained $pretrained --gpu $GPU

#infer
python ./scripts/train.py --image-root $image_root --snapshot $snapshot --backbone $backbone --gpu $GPU --infer

#eval
python ./scripts/eval.py --prediction-root $snapshot/pred_crf  --groundtruth-root $groundtruth_root