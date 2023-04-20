#!/bin/bash
cd /nfs/volume-92-1/shuweidong_i/the_mmdetection3d/mmdetection3d
python tools/train.py   configs/votenet/votenet_8x8_scannet-3d-18class.py   --work-dir  /nfs/volume-92-1/shuweidong_i/the_mmdetection3d/mmdetection3d/scripts/work_dirs

