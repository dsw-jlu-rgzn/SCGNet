# Semantic-Context Graph Network for Point-based 3D Object Detection (tcsvt)
Shuwei Dong, Xiaoyu Kong, Xingjia Pan, Fan Tang, Wei Li, Yi Chang, Weiming Dong

## Abstract

Point-based  indoor 3D object detection has received increasing attention with the large demand for augmented reality, autonomous driving, and robot technology in the industry. However, the detection precision suffers from inputs with semantic ambiguity, i.e., shape symmetries, occlusion, and texture missing, which would lead that different objects appearing similar from different viewpoints and then confusing the detection model. Typical point-based detectors relieve this problem via learning proposal representations with both geometric and semantic information, while the entangled representation may cause a reduction in both semantic and spatial discrimination. In this paper, we focus on alleviating the confusion from entanglement and then enhancing the proposal representation by considering the proposal's semantics and the context in one scene. A semantic-context graph network (SCGNet) is proposed, which mainly includes two modules: a category-aware proposal recoding module (CAPR) and a proposal context aggregation module (PCAg). To produce semantically clear features from entanglement representation, the CAPR module learns a high-level semantic embedding for each category to extract discriminative semantic clues. In view of further enhancing the proposal representation and leveraging the semantic clues, the PCAg module builds a graph to mine the most relevant context in the scene. With few bells and whistles, the SCGNet achieves SOTA performance and obtains consistent gains when applying to different backbones (0.9% ~ 2.4% on ScanNet V2 and 1.6% ~ 2.2% on SUN RGB-D for mAP@0.25).

 

##Notes
Our "Semantic-Context Graph Network for Point-based 3D Object Detection" was ACCEPTED as a Transactions Paper for publication in the IEEE Transactions on Circuits and Systems for Video Technology.


## Introduction

This repo is the official implementation of ["Semantic-Context Graph Network for Point-based 3D Object Detection"](http://ivc.ia.ac.cn/papers/SCGNet).

In this repository, we provide model implementation (with MMDetection3D V 0.17.1+da387db) as well as training scripts on ScanNet and SUN RGB-D.

## Results and models

### ScanNet V2

|Method | mAP@0.25 | mAP@0.5 |
|:---|:---:|:---:|
|[VoteNet*](https://arxiv.org/abs/1904.09664)       | 63.8 | 44.2 | 
|[VoteNet*](https://arxiv.org/abs/1904.09664)+SCGNet| 66.2 | 46.1 | 
|[H3DNet*](https://arxiv.org/abs/2006.05682)       | 66.1 | 47.7 | 
|[H3DNet*](https://arxiv.org/abs/2006.05682)+SCGNet | 67 | 49.3 | 
|[GroupFree3D*](https://arxiv.org/abs/2006.05682)(w2×,L12,0512) | 68.2 | 52.6 |
|[GroupFree3D*](https://arxiv.org/abs/2006.05682)(w2×,L12,0512)+SCGNet | 69.1 | 53.1 | 


### SUN RGB-D

|Method | mAP@0.25 | mAP@0.5 |
|:---|:---:|:---:|
|[VoteNet](https://arxiv.org/abs/1904.09664)*       | 59.1 | 35.8 |
|[VoteNet](https://arxiv.org/abs/1904.09664)*+SCGNet| 61.3 | 39.3 | 
|[imVoteNet](https://arxiv.org/abs/2001.10692)*| 64.0 | 37.8 |  
|[imVoteNet](https://arxiv.org/abs/2001.10692)*+SCGNet| 65.6 | 39.5 | 

**Notes:**

-  We use one NVIDIA NVIDIA Tesla P40 for training all models.
-  We report the best results on validation set during each training. 
-  \* denotes that the model is implemented on MMDetection3D.

## Install

This repo is built based on [MMDetection3D](V0.17.1), please follow the [getting_started.md](https://github.com/open-mmlab/mmdetection3ddocs/getting_started.md) for installation.

The code is tested under the following environment:
- GPU 0: Tesla P40
- PyTorch: 1.7.1+cu101
- C++ Version: 201402
- CUDA Runtime 10.1
- CuDNN 7.6.3
- Magma 2.5.2
- TorchVision: 0.8.2+cu101
- OpenCV: 4.5.1
- MMCV: 1.4.0
- MMCV Compiler: GCC 7.3
- MMCV CUDA Compiler: 10.1
- MMDetection: 2.14.0
- MMSegmentation: 0.16.0
- MMDetection3D: 0.17.1+da387db

## Data preparation

For SUN RGB-D, follow the [README](https://github.com/open-mmlab/mmdetection3d/data/sunrgbd/README.md) under the `/data/sunrgbd` folder.

For ScanNet, follow the [README](https://github.com/open-mmlab/mmdetection3d/data/scannet/README.md) under the `/data/scannet` folder.


## Training 

### ScanNet

For `VoteNet+SCGNet` training, run:

```bash
CUDA_VISIBLE_DEVICES=0 python tools/train.py configs/gcn/graph_votenet_iou.py
```

For `H3DNet+SCGNet` training, run:

```bash
CUDA_VISIBLE_DEVICES=0 python tools/train.py configs/gcn/h3dnet-graph-all-vote.py
```

```bash
CUDA_VISIBLE_DEVICES=0 python tools/train.py configs/gcn/h3dnet-graph-only-vote.py
```

For `GroupFree3D+SCGNet` training, run:

```bash
CUDA_VISIBLE_DEVICES=0 python tools/train.py configs/gcn/groupfree3d-L12O512w2x_graph_scannet.py
```


#### SUN RGB-D

For `VoteNet+SCGNet` training, run:

```bash
CUDA_VISIBLE_DEVICES=0 python tools/train.py configs/gcn/graph_votenet_sunrgbd_iou.py
```

For `imVoteNet+SCGNet` training, please go to the `mmdetection` dir and run:

```bash
CUDA_VISIBLE_DEVICES=0 python tools/train.py configs/gcn/graph_imvotenet-stage2_16x8_sunrgbd-3d-10class.py
```

### Interests
If you are interested in our work, please pay attention to the Wechat Official Accounts "计算创意与艺术"

