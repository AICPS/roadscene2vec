import sys, os
from pathlib import Path
sys.path.append(str(Path("../../")))
from sg2vec.data.preprocessor import Preprocessor as prepproc
import yaml
from sg2vec.data import dataset as ds
from abc import ABC
from pathlib import Path
from tqdm import tqdm
import torch
# import FileNotFoundError, ValueError #is this import needed?
import pickle as pkl
import ast
import json
from glob import glob
import cv2



from os import listdir
from os.path import isfile, join
from torch_geometric.data import Data, DataLoader, DataListLoader
import torchvision

"""RealPreprocessor takes in config and returns RawImageDataset object."""
class RealPreprocessor(prepproc):
    #TODO: RealPreprocessor preprocessor 
    def __init__(self,config):
        super(RealPreprocessor, self).__init__(config) 
        self.dataset = ds.RawImageDataset(self.conf)
        
    '''Extract scene data using raw images of each frame.'''
    def load(self):
        if not os.path.exists(self.dataset.dataset_path):
            raise FileNotFoundError(self.dataset.dataset_path)
        all_sequence_dirs = [x for x in Path(self.dataset.dataset_path).iterdir() if x.is_dir()]
        all_sequence_dirs = sorted(all_sequence_dirs, key=lambda x: int(x.stem.split('_')[0]))  
        self.dataset.folder_names = [path.stem for path in all_sequence_dirs]
        for path in tqdm(all_sequence_dirs):

            seq = int(path.stem.split('_')[0])
            self.dataset.action_types[seq] = path.stem.split('_')[2]
            label_path = (path/"label.txt").resolve()
            ignore_path = (path/"ignore.txt").resolve()
            
            if ignore_path.exists(): #record ignored sequences, and only load the sequences that were not ignored
                with open(str(path/"ignore.txt"), 'r') as label_f:
                    ignore_label = int(label_f.read())
                    if ignore_label:
                        self.dataset.ignore.append(seq)
                    else: #if don't want to ignore but file to ignore file exists
                        self.dataset.data[seq] = self._load_images(path)  
            else:#if don't want to ignore but path to file ignore file doesn't exist
                self.dataset.data[seq] = self._load_images(path)  
    
            if label_path.exists():
                with open(str(path/'label.txt'), 'r') as label_file:
                    lines = label_file.readlines()
                    l0 = 1.0 if float(lines[0].strip().split(",")[0]) >= 0 else 0.0 
                    self.dataset.labels[seq] = l0

    '''Represent each frame in sequence in terms of a tensor'''               
    def _load_images(self, path):

        raw_images_loc = (path/'raw_images').resolve()

        images = sorted([Path(f) for f in listdir(raw_images_loc) if isfile(join(raw_images_loc, f)) and ".DS_Store" not in f and "Thumbs" not in f], key = lambda x: int(x.stem.split(".")[0]))

        images = [join(raw_images_loc,i) for i in images] 

        sequence_tensor = {}
        shape = None
        num_of_frames_loaded = 0
        for image_path in images:
            frame_num = int(image_path.split('\\')[-1].split('.jpg')[0])
            if self.conf.output_format["color"] == "RGB":
                im = cv2.imread(str(image_path), 1) 
            elif self.conf.output_format["color"] == "Greyscale":
                im = cv2.imread(str(image_path), 0) 
            im = cv2.resize(im, (self.conf.output_format["width"], self.conf.output_format["width"]))
            if shape != None:
                if im.shape != shape:
                    raise ValueError("All images in a sequence must have the same shape")
            else:
                shape = im.shape
            if self.conf.output_format["color"] == "RGB":
                self.dataset.im_height, self.dataset.im_width, self.dataset.color_channels = im.shape
            elif self.conf.output_format["color"] == "Greyscale":
                self.dataset.im_height, self.dataset.im_width = im.shape
                self.dataset.color_channels = 0 #what should the color channel for greyscale be?
            sequence_tensor[frame_num] = im 
            if self.dataset.frame_limit != None:
                if self.dataset.frame_limit == len(sequence_tensor):
                    return sequence_tensor #return out if the frame limit has been met 
        return sequence_tensor
      
    '''Returns RawImageDataset object containing scengraphs, labels, and action types'''
    def getDataSet(self):
        return self.dataset
    
            
