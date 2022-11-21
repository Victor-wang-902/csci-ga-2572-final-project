# csci-ga-2572-final-project
Final competition code repository for CSCI-GA 2572 Deep Learning spring 2022.

Team Member:Xinhao Liu, Haresh Rengaraj R, Zecheng Wang

All codes in this repository were run on Greene.

## Pretraining
Perform Barlow twins pretraining of resnet50 inside Obj_SSL_barlow(from https://github.com/facebookresearch/barlowtwins)

1. Change the overlay path in `Obj_SSL_barlow/demo.slurm` to your corresponding path for environment and unlabeled dataset. 
2. run demo.slurm to train barlow twins for 150 epochs. Loss should drop to ~375 . The pretrained model will be available in Obj_SSL_barlow/checkpoint/checkpoint.pth

## Fine Tuning
We fine tuned the pretrained weights above by the folloing steps.
1. Change the path in line 59 in `fine_tune/barlowtwins.py` to the path of weight generated in the pretraining part
2. Change the overlay path in `fine_tune/run.sh` to your corresponding path for environment and labeled dataset. Then, change line 20 in `fine_tune/run.sh` into `python barlowtwins.py -n 20 --lr 0.0001` to train for 20 epochs with a learning rate of 0.0001. Use `sbatch run.sh` to start traning. This will give a mAP of approximately 0.20
3. Train another 20 epoch with learning rate = 5e-5, by changing line 20 in `fine_tun/run.sh` into `python barlowtwins_continue -n 20 --lr 0.00005`. This will give a mAP of approximately 0.28.

## Fine Tuning Transformers
Although our experiments with transformers ([DETR](https://github.com/facebookresearch/detr), [Swin](https://github.com/microsoft/Swin-Transformer)) was not very successful, this fork provides simplified codes and command for the experiments.
1. To finetune DETR, navigate to `detr`, install dependencies and run `./finetune_detr.sh`. The path of the pre-trained DETR (with [UP-DETR](https://github.com/dddzg/up-detr)) needs to be defined. In addition, finetuning can be done with [`mmdet`](https://github.com/open-mmlab/mmdetection). Navigate to `swin` and run `./fine_tune_detr.sh`. Note that the use of Anaconda is assumed and a new environment named `openmmlab` will be created for the command.
2. To finetune Swin Transformer, navigate to `swin`, and run `./fine_tune_swin_rcnn.sh`. Note that the use of Anaconda is assumed and a new environment named `openmmlab` will be created for the command.
