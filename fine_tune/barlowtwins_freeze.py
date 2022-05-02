# Feel free to modifiy this file. 
# It will only be used to verify the settings are correct 
# modified from https://pytorch.org/docs/stable/data.html

import os
import argparse

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

import transforms as T
import utils
from engine import train_one_epoch, evaluate


from dataset import UnlabeledDataset, LabeledDataset


parser = argparse.ArgumentParser()
parser.add_argument('-o', '--optimizer', type=str, default='adam', help='adam or sgd')
parser.add_argument('-n', '--n_epochs', type=int, default=30, help='number of epochs')
parser.add_argument('--lr',type=float,default='0.05',help='initial learning rate')
opt = parser.parse_args()


def get_transform(train):
    transforms = []
    transforms.append(T.ToTensor())
    if train:
        transforms.append(T.RandomHorizontalFlip(0.5))
    return T.Compose(transforms)

def get_model(num_classes):
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=False, pretrained_backbone=False)

    # get number of input features for the classifier
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    # replace the pre-trained head with a new one
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

    return model

def main():
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    num_classes = 101
    train_dataset = LabeledDataset(root='/labeled', split="training", transforms=get_transform(train=True))
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=16, shuffle=True, num_workers=16, collate_fn=utils.collate_fn)

    valid_dataset = LabeledDataset(root='/labeled', split="validation", transforms=get_transform(train=False))
    valid_loader = torch.utils.data.DataLoader(valid_dataset, batch_size=16, shuffle=False, num_workers=16, collate_fn=utils.collate_fn)

    model = get_model(num_classes)
    model.to(device)

    # state_dict = torch.load("/scratch/xl3136/dl-sp22-final-project/Obj_SSL_barlow/checkpoint/checkpoint2.pth")
    # new_state_dict = {}
    # for key in state_dict['model'].keys():
    #     if key.startswith('module.backbone'):
    #         new_key = key.replace('module.backbone', 'backbone.body')
    #         new_state_dict[new_key] = state_dict['model'][key]
    # missing_keys, unexpected_keys = model.load_state_dict(new_state_dict, strict=False)
    state_dict = torch.load('check_point_bt2adam0.0001.pth')
    missing_keys, unexpected_keys = model.load_state_dict(state_dict)
    print("missing keys:")
    print(missing_keys)
    print("unexpected_keys:")
    print(unexpected_keys)
    print("loaded successfully")
    assert()

    params = [p for p in model.parameters() if p.requires_grad]
    if opt.optimizer == 'sgd':
        optimizer = torch.optim.SGD(params, lr=opt.lr, momentum=0.9, weight_decay=0.005)
        lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)
    elif opt.optimizer == 'adam':
        optimizer = torch.optim.Adam(params, lr=opt.lr)
    else:
        assert False, "Optimizer have to be sgd or adam"

    num_epochs = opt.n_epochs
    print("start training")
    evaluate(model, valid_loader, device=device)
    for epoch in range(num_epochs):
        # train for one epoch, printing every 10 iterations
        train_one_epoch(model, optimizer, train_loader, device, epoch, print_freq=100)
        # update the learning rate
        # lr_scheduler.step()
        # evaluate on the test dataset
        evaluate(model, valid_loader, device=device)
        # save check point
        save_name = "check_point_bt2c"+str(opt.lr)+".pth"
        torch.save(model.state_dict(), save_name)

    print("That's it!")

if __name__ == "__main__":
    main()
