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

python driver_detr.py configs/detr/detr_r50_8x2_150e_coco.py