import os
import sys
from pathlib import Path

sys.path.append(str(Path("../../../")))
from preprocessor import Preprocessor as prepproc
from roadscene2vec.data import dataset as ds
from pathlib import Path
from tqdm import tqdm
import cv2
from os import listdir
from os.path import isfile, join


"""RealPreprocessor takes in config and returns RawImageDataset object."""
class RealPreprocessor(prepproc):
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
            label_path = (path/"label.txt").resolve()
            ignore_path = (path/"ignore.txt").resolve()
            
            if ignore_path.exists(): #record ignored sequences, and only load the sequences that were not ignored
                with open(str(path/"ignore.txt"), 'r') as label_f:
                    ignore_label = int(label_f.read())
                    if ignore_label:
                        self.dataset.ignore.append(seq)
                        continue #skip to next seq if ignore path exists

            self.dataset.data[seq] = self._load_images(path)
            self.dataset.action_types[seq] = "lanechange"
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
        modulo = 0
        acc_number = 0
        if(self.dataset.frame_limit != None):
            modulo = int(len(images) / self.dataset.frame_limit)  #subsample to frame_limit 
        if(self.dataset.frame_limit == None or modulo == 0):
            modulo = 1

        self.dataset.im_height, self.dataset.im_width = self.conf.output_format["height"], self.conf.output_format["width"]
        if self.conf.output_format["color"] == "RGB":
            self.dataset.color_channels = 3
        elif self.conf.output_format["color"] == "Grayscale":
            self.dataset.color_channels = 1

        for i in range(0, len(images)):
            if (i % modulo == 0 and self.dataset.frame_limit == None) or (i % modulo == 0 and acc_number < self.dataset.frame_limit):
                image_path = images[i]
                frame_num = int(Path(image_path).stem)
                if self.conf.output_format["color"] == "RGB":
                    im = cv2.imread(str(image_path), cv2.IMREAD_COLOR) 
                elif self.conf.output_format["color"] == "Greyscale":
                    im = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE) 
                im = cv2.resize(im, (self.dataset.im_width, self.dataset.im_height)).transpose(2, 0, 1) #convert to (channels, height, width) format
                if shape != None:
                    if im.shape != shape:
                        raise ValueError("All images in a sequence must have the same shape")
                else:
                    shape = im.shape
                sequence_tensor[frame_num] = im 
                acc_number += 1
        return sequence_tensor
      
    '''Returns RawImageDataset object containing scengraphs, labels, and action types'''
    def getDataSet(self):
        return self.dataset
    
            
