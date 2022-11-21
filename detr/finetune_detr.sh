#!/bin/bash

python main.py --batch_size 2 --lr 1e-5 --lr_backbone 1e-6 --epochs 150 --world_size 1 --num_workers 8 --bbox_loss_coef 8 --giou_loss_coef 4 --eos_coef 1  --output_dir detr --resume checkpoint/updetr.pth