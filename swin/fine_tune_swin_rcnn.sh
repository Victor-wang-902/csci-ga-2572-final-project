#!/bin/bash

conda create -n openmmlab python=3.7 pytorch==1.6.0 cudatoolkit=10.1 torchvision -c pytorch -y
conda activate openmmlab
pip install openmim
mim install mmcv-full
git clone https://github.com/open-mmlab/mmdetection.git
cd mmdetection
pip install -r requirements/build.txt
pip install -v -e .
conda activate openmmlab

python driver.py configs/faster_rcnn/faster_rcnn_swin_fpn_1x_coco_stretch.py --work-dir=swin_new_test_3
