# P2T
from __future__ import print_function
import argparse

import torch
from torch.utils.data import DataLoader
import time
import math
import sys 
sys.path.append('C:\\Users\\NGN\\dev\\NTPN\\models')
sys.path.append('C:\\Users\\NGN\\dev\\NTPN\\datasets')
from mtp3 import MTP
from helper3 import PredictHelper
from backbone_n import ResNeXtBackbone, DenseNetBackbone, WideResNetBackbone, ResNetBackbone, MobileNetBackbone
from nuscenes.prediction.models.mtp import MTPLoss
import yaml
from torch.utils.tensorboard import SummaryWriter
import numpy as np



# Read config file
config_file = './configs/ns.yml'
with open(config_file, 'r') as yaml_file:
    config = yaml.safe_load(yaml_file)

# Import dataset
if config['ds'] == 'ns':
    from ns import NS as DS

# Initialize device:
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
# Tensorboard summary writer:
writer = SummaryWriter(log_dir=config['opt_mtp']['log_dir']) # log_dir: 'checkpts/ns/loss/log'

# Initialize datasets:
tr_set = DS(config['dataroot'],
            config['train'],
            # t_f=config['t_f']
            )

val_set = DS(config['dataroot'],
             config['val'],
             # t_f=config['t_f'],
             )
print(f"len(tr_set) : {len(tr_set)}") # 742
print(f"len(val_set) : {len(val_set)}") # 61
# Initialize data loaders:
tr_collate_fn = None
tr_dl = DataLoader(tr_set,
                   batch_size=config['opt_mtp']['batch_size'], # 4
                   shuffle=True,
                   num_workers=config['num_workers'],
                   collate_fn=tr_collate_fn)

val_collate_fn = None
val_dl = DataLoader(val_set,
                    batch_size=config['opt_mtp']['batch_size'], # 4
                    shuffle=True,
                    num_workers=config['num_workers'],
                    collate_fn=val_collate_fn)

# Initialize Models:
parser = argparse.ArgumentParser()
parser.add_argument("-m", "--model", default="0", type=int)
args = parser.parse_args()
idx_bb = args.model



if idx_bb==0: backbone = ResNetBackbone('resnet50')
elif idx_bb==1: backbone = ResNeXtBackbone('resnext50_32x4d')
elif idx_bb==2: backbone = WideResNetBackbone('wide_resnet50_2') # gpu memory issue
elif idx_bb==3: backbone = DenseNetBackbone('densenet161')  # # gpu memory issue
elif idx_bb==4: backbone = ResNetBackbone('resnet18')
elif idx_bb==5: backbone = MobileNetBackbone('mobilenet_v2')
else:
    backbone = None
    print("backbone error!")

print(f'selected model: {idx_bb}:{backbone}')
net = MTP(backbone, num_modes=2).float()
net = net.to(device)
loss_function = MTPLoss(num_modes=2, regression_loss_weight=1., angle_threshold_degrees=5.)

# Initialize Optimizer:
num_epochs = config['opt_mtp']['num_epochs'] # 10
optimizer = torch.optim.Adam(net.parameters(), lr=config['opt_mtp']['lr']) # lr=0.0001

# Load checkpoint if specified in config:
if config['opt_mtp']['load_checkpt']: # default : false
    print(f"load checkpt:{config['opt_mtp']['load_checkpt']}")
    checkpoint = torch.load(config['opt_mtp']['checkpt_path'])
    net.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    start_epoch = checkpoint['epoch'] + 1
    val_loss = checkpoint['loss']
    min_val_loss = checkpoint['min_val_loss']
else:
    print('default start!!!')
    start_epoch = 1
    val_loss = math.inf
    # val_loss = 10000
    min_val_loss = math.inf
    # min_val_loss = 0.56234

print(f"start_epoch: {start_epoch}, val_loss:{val_loss}, min_val_loss:{min_val_loss}")


# ======================================================================================================================
# Main Loop
# ======================================================================================================================

# Forever increasing counter to keep track of iterations (for tensorboard log).
iters_epoch = len(tr_set) // config['opt_mtp']['batch_size'] # batch_size: 8 # train : 32186 # mini_train : 742
iters = (start_epoch - 1) * iters_epoch


print('Start Training...')
for epoch in range(start_epoch, start_epoch + num_epochs):

    # __________________________________________________________________________________________________________________
    # Train
    # __________________________________________________________________________________________________________________

    # Set batchnorm layers to train mode
    net.train()

    # Variables to track training performance
    tr_loss_path = 0


    tr_time = 0 # time of repetition

    # For tracking training time
    st_time = time.time() #

    # Load batch
    for i, data in enumerate(tr_dl):

        # Process inputs
        img_tensor, agent_state_vector, ground_truth, _, _, _ = data
        
        img_tensor = img_tensor.float().to(device)
        agent_state_vector = agent_state_vector.float().to(device)
        ground_truth = ground_truth.float().to(device) # ([batch_size, 1, 12, 2])

        
        # Forward pass
        predictions = net(img_tensor, agent_state_vector) # 1

        # Compute loss
        loss = loss_function(predictions, ground_truth) # 2 # tensor(14.2326, device='cuda:0', grad_fn=<MeanBackward0>)
        current_loss = loss.cpu().detach().numpy()

        print(f"Current loss is {current_loss:.4f}") # 14.23262 # float32

        # Backprop
        optimizer.zero_grad()  # 3
        loss.backward() #
        optimizer.step() # 5

        # if np.allclose(current_loss, min_val_loss, atol=1e-4):
        #     print(f"Achieved near-zero loss after {iters} iterations.")
        #     break


        # Keep time
        minibatch_time = time.time() - st_time
        tr_loss_path += loss
        tr_time += minibatch_time
        st_time = time.time()

        # Tensorboard train metrics
        writer.add_scalar('train/loss (paths)', loss, iters) # add_scalar(tag, scalar_value, global_step=None, walltime=None)

        # Increment global iteration counter for tensorboard
        iters += 1 # 

        # Print/log train loss (current_loss) and ETA for epoch after pre-defined steps
        iters_log = config['opt_mtp']['steps_to_log_train_loss'] # steps_to_log_train_loss: 100
        if i % iters_log == iters_log - 1:
            eta = tr_time / iters_log * (len(tr_set) / config['opt_mtp']['batch_size'] - i)
            print("Epoch no:", epoch,
                  "| Epoch progress(%):", format(i / (len(tr_set) / config['opt_mtp']['batch_size']) * 100, '0.2f'),
                  "| Train loss (paths):", format(tr_loss_path / iters_log, '0.5f'),
                  "| Val loss prev epoch", format(val_loss, '0.7f'),
                  "| Min val loss", format(min_val_loss, '0.5f'),
                  "| ETA(s):", int(eta))

            # Reset variables to track training performance
            tr_loss_path = 0
            tr_time = 0
    # writer.close()
 # __________________________________________________________________________________________________________________
    # Validate
    # __________________________________________________________________________________________________________________
    print('Calculating validation loss...')

    # Set batchnorm layers to eval mode, stop tracking gradients
    net.eval()
    with torch.no_grad(): ##

        # Variables to track validation performance
        val_loss_path = 0
        val_batch_count = 0

        # Load batch
        for k, data_val in enumerate(val_dl):

            # Process inputs
            img_tensor, agent_state_vector, ground_truth, _, _, _ = data_val

            img_tensor = img_tensor.float().to(device)
            agent_state_vector = agent_state_vector.float().to(device)
            ground_truth = ground_truth.float().to(device)

            #  Calculate predictions, loss
            predictions = net(img_tensor, agent_state_vector)
            loss = loss_function(predictions, ground_truth)


            val_loss_path += loss
            val_batch_count += 1


    # Print validation losses
    print('Val loss (paths) :', format(val_loss_path / val_batch_count, '0.5f'))
    val_loss = val_loss_path / val_batch_count

    # Tensorboard val metrics
    writer.add_scalar('val/loss_paths', val_loss_path / val_batch_count, iters)
    writer.flush()

    # Save checkpoint
    if config['opt_mtp']['save_checkpoints']: # default:true
        model_path = config['opt_mtp']['checkpt_dir'] + '/' + str(epoch) + '.tar' # 'checkpts/ns/loss/weights'
        torch.save({
            'epoch': epoch,
            'model_state_dict': net.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': val_loss,
            'min_val_loss': min(val_loss, min_val_loss)
        }, model_path)

    # Save best model if applicable
    if val_loss < min_val_loss:
        min_val_loss = val_loss
        model_path = config['opt_mtp']['checkpt_dir'] + '/best.tar' # 'checkpts/ns/loss/weights'
        torch.save({
            'epoch': epoch,
            'model_state_dict': net.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': val_loss,
            'min_val_loss': min_val_loss
        }, model_path)


# Close tensorboard writer
writer.close()