from torch.utils.data import Dataset
import numpy as np
import torchvision.transforms as tsfm
import torch
from nuscenes import NuScenes
import sys
sys.path.append('C:\\Users\\NGN\\dev\\NTPN\\models')
from helper3 import PredictHelper
from nuscenes.prediction.input_representation.static_layers import StaticLayerRasterizer
from nuscenes.prediction.input_representation.agents import AgentBoxesWithFadedHistory
from nuscenes.prediction.input_representation.interface import InputRepresentation, AgentRepresentation
from nuscenes.prediction.input_representation.combinators import Rasterizer
from nuscenes.prediction.input_representation.utils import convert_to_pixel_coords, get_crops, get_rotation_matrix
# from nuscenes.eval.prediction.splits import get_prediction_challenge_split
from splits import get_prediction_challenge_split2
from nuscenes.prediction.helper import quaternion_yaw
from pyquaternion import Quaternion
from PIL import Image
import cv2
import torch.nn.functional as f
from typing import Any, Dict, List, Tuple
import os

class NS(Dataset):

    def __init__(self,
                 dataroot: str,
                 split: str,
                 t_f: float = 6,
                 ):
        """
        Initializes dataset class for nuScenes prediction
        :param dataroot: Path to tables and data
        :param split: Dataset split for prediction benchmark ('mini_train'/'mini_val'/'mini_test' or 'train'/'train_val'/'val')
        :param t_f: Prediction horizon in seconds
        """

        # Nuscenes dataset and predict helper
        self.dataroot = dataroot
        self.ns = NuScenes('v1.0-trainval', dataroot=dataroot)
        # self.ns = NuScenes('v1.0-mini', dataroot=dataroot)
        self.helper = PredictHelper(self.ns)
        self.token_list = get_prediction_challenge_split2(split, dataroot=dataroot)
        self.static_layer_rasterizer = StaticLayerRasterizer(self.helper)
        self.agent_rasterizer = AgentBoxesWithFadedHistory(self.helper, seconds_of_history=1)
        self.mtp_input_representation = InputRepresentation(self.static_layer_rasterizer, self.agent_rasterizer, Rasterizer())

        # Useful parameters
        self.t_f = t_f


    def __len__(self):
        return len(self.token_list)

    def __getitem__(self, idx):
        """
        Returns inputs, ground truth values and other utilities for data point at given index
        :return image_tensor : rasterized 3*500*500 image to tensor
        :return agent_state_vector : vel, acc, heading_change_rate
        :return ground_truth: ground truth future trajectory, default 6s at 0.5 Hz sampling frequency
        :return instance_token: nuScenes instance token for prediction instance
        :return sample_token: nuScenes sample token for prediction instance
        :return idx: instance id (mainly for debugging)
        """

        # Nuscenes instance and sample token for prediction data point
        instance_token, sample_token = self.token_list[idx].split("_")


        # When dataset is being used for training/validation/testing :

        # Get ground truth future for agent:
        fut = self.helper.get_future_for_agent(instance_token,
                                                sample_token,
                                                seconds=self.t_f,
                                                in_agent_frame=True)
        fut = torch.from_numpy(fut) # [12, 2]
        ground_truth = fut.reshape(1, 12, 2)
        ground_truth = ground_truth.float()
        
        # Get image_tensor
        anns = [ann for ann in self.ns.sample_annotation if ann['instance_token'] == instance_token]
        img = self.mtp_input_representation.make_input_representation(instance_token, sample_token)
        image_tensor = torch.Tensor(img).permute(2, 0, 1)
        
        # Get agent_state_vector
        agent_state_vector = torch.Tensor([self.helper.get_velocity_for_agent(instance_token, sample_token),
                                self.helper.get_acceleration_for_agent(instance_token, sample_token),
                                self.helper.get_heading_change_rate_for_agent(instance_token, sample_token)])

        return image_tensor, agent_state_vector, ground_truth, instance_token, sample_token, idx



    

   



