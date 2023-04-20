#!/bin/bash
cd /nfs/volume-92-1/shuweidong_i/the_mmdetection3d/mmdetection3d

CUDA_VISIBLE_DEVICES=0,1,2,3 PORT=29503  ./tools/dist_train.sh ./configs/votenet/gvotenet_global_resisual_8x8_scannet-3d-18-class.py   4  --work-dir /nfs/volume-92-1/shuweidong_i/the_mmdetection3d/mmdetection3d/scripts/work_dirs




