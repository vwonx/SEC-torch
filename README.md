# Paper: Seed, Expand, Constrain: Three Principles for Weakly-Supervised Image Segmentation

PyTorch implementation of "Seed, Expand, Constrain: Three Principles for Weakly-Supervised Image Segmentation", ECCV2016

Thanks to the work of [halbielee](https://github.com/halbielee), the code of this repository borrow heavly from his [SEC_pytorch](https://github.com/halbielee/SEC_pytorch) repository. 

- Paper : [https://arxiv.org/abs/1603.06098](https://arxiv.org/abs/1603.06098)

- Official code : [caffe implmentation](https://github.com/kolesman/SEC)


## Preparation
For using this code, you have to do something else:

### 1. Requirement
- NVIDIA GPU
- Python 3.6
- PyTorch
- opencv-python (OpenCV for Python)
- Numpy
- Fully connected CRF wrapper (requires the Eigen3 package)
```bash
pip install CRF/
```

### 2. Download the data & pretrain-model
- Download the VOC 2012 dataset (need include SegmentationClassAug), and config the path accordingly in the run.sh
- Download the pretrained model from [here](https://drive.google.com/a/yonsei.ac.kr/file/d/1Fs25jmy9uZJxLlWFfdOESOs5EXnAr0OK/view?usp=sharing) and put it into the folder SEC-torch/data/pretrained

### 3. Prepare seed file
```bash
cd SEC-torch

gzip -kd data/seed/localization_cues.pickle.gz
```

## Execution
```bash
chmod +x run.sh
./run.sh
```
This script will automatically run the training, testing (on val set) and save checkpoing in folder SEC-torch/snapshot
