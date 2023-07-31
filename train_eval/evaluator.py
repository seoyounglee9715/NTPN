from __future__ import print_function
import argparse
import sys
import torch
from torch.utils.data import DataLoader

sys.path.append('C:\\Users\\NGN\\dev\\NTPN\\models')
sys.path.append('C:\\Users\\NGN\\dev\\NTPN\\datasets')

from mtp3 import MTP
from helper3 import PredictHelper
from backbone_n import ResNeXtBackbone, DenseNetBackbone, WideResNetBackbone, ResNetBackbone, MobileNetBackbone
from nuscenes.prediction.models.mtp import MTPLoss

import yaml

import numpy as np
from ns import NS as DS

import json
from nuscenes.eval.prediction.config import PredictionConfig
from nuscenes.prediction.helper import convert_local_coords_to_global
from nuscenes.eval.prediction.data_classes import Prediction
# from nuscenes.eval.prediction.compute_metrics import compute_metrics
sys.path.append('C:\\Users\\NGN\\dev\\NTPN\\utils')
from compute_metrics import compute_metrics
import os
import time


start=time.time()
# Read config file
config_file = 'configs/ns.yml'
with open(config_file, 'r') as yaml_file:
    config = yaml.safe_load(yaml_file)

# Initialize device:
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Initialize dataset:
ts_set = DS(config['dataroot_t'], # 'E:\\NuScenes\\trainval\\data\\sets\\nuscenes'
            config['test'],
            # t_f=config['t_f'],
            )

# Initialize data loader:
ts_dl = DataLoader(ts_set,
                   batch_size=4,
                   shuffle=True,
                   num_workers=config['num_workers'])

print(f"len(ts_set) : {len(ts_set)}") #


# cnn backbone select
parser = argparse.ArgumentParser()
parser.add_argument("-m", "--model", default="0", type=str) # default: resNet50
args = parser.parse_args()
idx_bb = args.model

if idx_bb=='0': backbone = ResNetBackbone('resnet50')
elif idx_bb=='1': backbone = ResNeXtBackbone('resnext50_32x4d')
elif idx_bb=='2': backbone = WideResNetBackbone('wide_resnet50_2') # gpu memory issue
elif idx_bb=='3': backbone = DenseNetBackbone('densenet161')  # # gpu memory issue
elif idx_bb=='4': backbone = ResNetBackbone('resnet18')
elif idx_bb=='5': backbone = MobileNetBackbone('mobilenet_v2')
else:
    backbone = None
    print("backbone error!")

print(f'selected model: {idx_bb}:{backbone}')

# Initialize Models:
net = MTP(backbone, num_modes=2).float().to(device)
# s_d = net.load_state_dict(torch.load(config['opt_mtp']['checkpt_dir'] + '/' + idx_bb + '/'+'best.tar'))
net.load_state_dict(torch.load(config['opt_mtp']['checkpt_dir'] + '/' + idx_bb + '/'+'best.tar')['model_state_dict']) # 'checkpts/ns/loss/weights/1/best.tar'
# print(f' state dict of model:{s_d}')
for param in net.parameters():
    param.requires_grad = False
net.eval()

loss_function = MTPLoss(num_modes=2, regression_loss_weight=1., angle_threshold_degrees=5.)


# Prediction helper and configs:
helper = ts_set.helper

with open('configs/predict_2020_icra.json', 'r') as f:
    pred_config = json.load(f)
pred_config_2020 = PredictionConfig.deserialize(pred_config, helper)


# Lists of predictions
mtp_preds = []
cnt = 0
# Load batch

for i, data in enumerate(ts_dl):
    j = 0
    # Process inputs
    img_tensor, agent_state_vector, ground_truth, instance_token, sample_token, _ = data

    img_tensor = img_tensor.float().to(device)
    agent_state_vector = agent_state_vector.float().to(device)
    ground_truth = ground_truth.float().to(device)  # ([batch_size, 1, 12, 2])

    # Forward pass
    # predictions = net(img_tensor, agent_state_vector)  # 1

    # Compute loss
    # loss = loss_function(predictions, ground_truth)  # 2 # tensor(14.2326, device='cuda:0', grad_fn=<MeanBackward0>)
    # current_loss = loss.cpu().detach().numpy()

    # Generate trajectories
    pred = net(img_tensor, agent_state_vector)

    t_pred = pred[0][:48]
    p_pred = pred[0][48:]

    if i != 2260:
        t_pred1 = pred[1][:48]
        p_pred1 = pred[1][48:]
        t_pred2 = pred[2][:48]
        p_pred2 = pred[2][48:]
        t_pred3 = pred[3][:48]
        p_pred3 = pred[3][48:]

    t_np = t_pred.reshape(2, 12, 2).detach().cpu().numpy()  # trajectory numpy
    p_np = p_pred.detach().cpu().numpy()  # probability numpy

    if i != 2260:
        t_np1 = t_pred1.reshape(2, 12, 2).detach().cpu().numpy()
        p_np1 = p_pred1.detach().cpu().numpy()
        t_np2 = t_pred2.reshape(2, 12, 2).detach().cpu().numpy()
        p_np2 = p_pred2.detach().cpu().numpy()
        t_np3 = t_pred3.reshape(2, 12, 2).detach().cpu().numpy()
        p_np3 = p_pred3.detach().cpu().numpy()

    tmp = Prediction(instance_token[0], sample_token[0], prediction=t_np, probabilities=p_np) # instance_token type: tutple
    mtp_preds.append(tmp.serialize())

    if i != 2260:
        tmp1 = Prediction(instance_token[1], sample_token[1], prediction=t_np1, probabilities=p_np1)
        mtp_preds.append(tmp1.serialize())
        tmp2 = Prediction(instance_token[2], sample_token[2], prediction=t_np2, probabilities=p_np2)
        mtp_preds.append(tmp2.serialize())
        tmp3 = Prediction(instance_token[3], sample_token[3], prediction=t_np3, probabilities=p_np3)
        mtp_preds.append(tmp3.serialize())

    cnt += 1
    print(cnt)
    # output 파일 저장명
    json.dump(mtp_preds, open(os.path.join(config['pred_output_dir']  # 'E:/NuScenes/trainval/data/sets/nuscenes/output2'
                                           + '/' +idx_bb + 'pred.json'), "w"))

results = compute_metrics(mtp_preds, helper, pred_config_2020)
json.dump(results, open(config['pred_output_dir']
                        + '/' +idx_bb +'pred.json'
                        .replace('.json', '_metrics.json'), "w"), indent=2)
print('Results : \n' + str(results))

end=time.time()
duration=end-start
print(duration)

