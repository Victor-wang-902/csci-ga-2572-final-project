import torch
from demo import *

state_dict = torch.load('/scratch/xl3136/dl-sp22-final-project/Obj_SSL_barlow/checkpoint/checkpoint.pth', map_location=torch.device('cpu'))
model = get_model(100)

new_state_dict = {}
for key in state_dict['model'].keys():
    if key.startswith('module.backbone'):
        new_key = key.replace('module.backbone', 'backbone.body')
        new_state_dict[new_key] = state_dict['model'][key]

# print(state_dict['model'].keys())
# print()
# print(new_state_dict.keys())
model.load_state_dict(new_state_dict)