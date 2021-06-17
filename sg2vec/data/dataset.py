### keep for developement.
# import sys, os
# sys.path.append(os.path.dirname(sys.path[0]))
from abc import ABC
# from pathlib import Path
# from tqdm import tqdm
# import torch
# import FileNotFoundError, ValueError #is this import needed?
import pickle as pkl
# import ast
# import json
# from glob import glob
# import cv2

# from os import listdir
# from os.path import isfile, join
# import pdb

import pathlib
import platform

#sg2vec/data/dataset.py

'''
Abstract class defining dataset properties and functions

Datasets must be structured as follows:
# dataset_path / <sequence_id> / raw_images / <image files> (sorted in ascending filename order.)
# dataset_path / <sequence_id> / gt_data / <ground truth data files> (sorted in ascending filename order.)
# dataset_path / <sequence_id> / label.txt (sorted ascending filename order or simply one for entire sequence.)
# dataset_path / <sequence_id> / metadata.txt (sorted in ascending filename order or one for the entire sequence.)

All directories under dataset_path will be considered to be sequences containing data and labels.

The resulting RawImageDataset will be stored in the following location:
# dataset_path / <image_dataset_path>.pkl

The resulting SceneGraphDataset will be stored in the following location:
# dataset_path / <sg_dataset_path>.pkl

'''
class BaseDataset(ABC):
    def __init__(self, config):
        self.dataset_path = config.location_data["input_path"]
        self.config = config
        self.data = None
        self.labels = None
        self.meta = None
#         self.dataset_type = None
        self.dataset_save_path = config.location_data["data_save_path"]
        self.dataset_type = config.dataset_type
        self.action_types = None
        self.raw_scenes = None
        self.ignore = None
        self.folder_names = None


    #load/save data from dataset_path into data, labels, meta
    def save(self):
        with open(self.dataset_save_path, 'wb') as f:
            pkl.dump(self, f, fix_imports=False)

    def load(self):
        with open(self.dataset_save_path, 'rb') as f:
            pathlib.WindowsPath = pathlib.PosixPath
            return pkl.load(f, fix_imports=False)


'''
Dataset containing ground truth information about the road state.
'''
class GroundTruthDataset(BaseDataset):
    # REQ: contains ground truth information about objects in the scene
    def __init__(self, config):
        super(GroundTruthDataset, self).__init__(config)
        
        
        #indexes for these three below are seq numbers
        self.meta = {}
        self.action_types = {}
        self.labels = {} 
        
        self.raw_scenes = {} #this is dict within dict: raw_scenes[sequence_number][frame_number]


'''
Dataset containing image data and associated information only.
'''
class RawImageDataset(BaseDataset):
    # REQ: the dataset that only contains raw images
    # REQ: this dataset can be used for scene graph extractor
    # REQ: this dataset can be used by CNN-based approach directly.
    def __init__(self, config = None):
        if config != None:
            super(RawImageDataset, self).__init__(config)
            self.im_height = None
            self.im_width =  None
            self.color_channels =  None
            self.frame_limit = config.frame_data["frames_limit"]
            self.dataset_type = 'image'
            self.data = {}   #{sequence{frame{frame_data}}} 
            self.labels = {} #{sequence{label}}
            self.action_types = {} #{sequence{action}}
            self.ignore = [] #sequences to ignore


'''
Dataset containing scene-graph representations of the road scenes.
This dataset is generated by the scene graph extractors and saved as a pkl file.
'''
class SceneGraphDataset(BaseDataset):
    # REQ: the dataset that only contains scene-graphs
    # meta data dict
    #action types dict
    # labels' dict
    # should be able to be converted into graph dataset or sequenced graph dataset.
    def __init__(self, config = None, scene_graphs= None, action_types= None, label_data= None,meta_data = None):
        if config != None:
            super(SceneGraphDataset, self).__init__(config)
            self.dataset_type = 'scenegraph'
            self.scene_graphs = scene_graphs
            self.meta = meta_data
            self.labels = label_data
            self.action_types = action_types

