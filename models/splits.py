# nuScenes dev-kit.
# Code written by Freddy Boulton.

import json
import os
from itertools import chain
from typing import List

import sys
# from nuscenes.utils.splits import create_splits_scenes
sys.path.append('C:\\Users\\NGN\\dev\\NTPN\\utils')
from splits_u import create_splits_scenes


NUM_IN_TRAIN_VAL = 200


def get_prediction_challenge_split2(split: str, dataroot: str = '/data/sets/nuscenes') -> List[str]:
    """
    Gets a list of {instance_token}_{sample_token} strings for each split.
    :param split: One of 'mini_train', 'mini_val', 'train', 'val'.
    :param dataroot: Path to the nuScenes dataset.
    :return: List of tokens belonging to the split. Format {instance_token}_{sample_token}.
    """
    if split not in {'mini_train', 'mini_val', 'mini_test','train', 'train_val', 'val'}:
        raise ValueError("split must be one of (mini_train, mini_val, 'mini_test', train, train_val, val)")
    
    if split == 'train_val':
        split_name = 'train'
    else:
        split_name = split

    path_to_file = os.path.join(dataroot, "maps", "prediction", "prediction_scenes.json")
    prediction_scenes = json.load(open(path_to_file, "r"))
    scenes = create_splits_scenes()
    # print(f"len(scenes): {len(scenes)}")
    # print(scenes)
    scenes_for_split = scenes[split_name]
    print(f"len(scenes_for_split1): {len(scenes_for_split)}")
    print(scenes_for_split)
    
    if split == 'train':
        scenes_for_split = scenes_for_split[NUM_IN_TRAIN_VAL:]
        print(f"len(scenes_for_split2): {len(scenes_for_split)}")
        print(scenes_for_split)
    if split == 'train_val':
        scenes_for_split = scenes_for_split[:NUM_IN_TRAIN_VAL]
        print(f"len(scenes_for_split2): {len(scenes_for_split)}")
        print(scenes_for_split)

    token_list_for_scenes = map(lambda scene: prediction_scenes.get(scene, []), scenes_for_split)

    return list(chain.from_iterable(token_list_for_scenes))
