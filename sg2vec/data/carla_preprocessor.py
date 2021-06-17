from  .preprocessor import Preprocessor as prepproc
import yaml
import data.dataset as ds
import sys, os
sys.path.append(os.path.dirname(sys.path[0]))
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




"""CarlaPreprocessor takes in config and returns GroundTruthDataset object."""
class CarlaPreprocessor(prepproc):
    def __init__(self,config):
        super(CarlaPreprocessor, self).__init__(config)
        self.dataset = ds.GroundTruthDataset(self.conf)
    
    '''Extract scene data using raw data json files which contain data for each frame in the sequence'''    
    def load(self):
        if not os.path.exists(self.dataset.dataset_path):
            raise FileNotFoundError(self.dataset.dataset_path)
        
        all_sequence_dirs = [x for x in Path(self.dataset.dataset_path).iterdir() if x.is_dir()]
        all_sequence_dirs = sorted(all_sequence_dirs, key=lambda x: int(x.stem.split('_')[0]))  
        self.dataset.folder_names = [path.stem for path in all_sequence_dirs]

        
   
        for path in tqdm(all_sequence_dirs):
            seq = int(path.stem.split('_')[0])
            self.dataset.action_types[seq] = path.stem.split('_')[1]
            label_path = (path/"label.txt").resolve()
            metadata_path = (path/"metadata.txt").resolve()

            if not label_path.exists():
                raise FileNotFoundError((path/'label.txt').resolve())
            else:
                with open(str(path/'label.txt'), 'r') as label_file:
                    lines = label_file.readlines()
                    l0 = 1.0 if float(lines[0].strip().split(",")[0]) >= 0 else 0.0 #is this correct or shld i just use raw values?
                    self.dataset.labels[seq] = l0 

              
            if not metadata_path.exists():
                raise FileNotFoundError((path/'metadata.txt').resolve())
            else:
                with open(str(path/'metadata.txt'), 'r') as md_file:
                    md = md_file.read()
                    self.dataset.meta[seq] = ast.literal_eval(md)
                  
            txt_path = sorted(list(glob("%s/**/*.json" % str(path/"scene_raw"), recursive=True)))[0]
            with open(txt_path, 'r') as scene_dict_f:
                try:
                    self.dataset.raw_scenes[seq] = json.loads(scene_dict_f.read())
            
                except Exception as e:
                    import traceback
                    print("We have problem parsing the dict.json in %s"%txt_path)
                    print(e)
                    traceback.print_exc()

                                           
    
    '''Returns GroundTruthDataset object containing scengraphs, labels, action types, and meta data'''
    def getDataSet(self):
        return self.dataset
                
