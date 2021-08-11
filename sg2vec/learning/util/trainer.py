import sys, os, pdb
from pathlib import Path
sys.path.append(str(Path("../../")))
import torch
import torch.optim as optim
import torch.nn as nn
import numpy as np
import pandas as pd
import random
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, precision_score, recall_score, roc_auc_score, roc_curve
from sklearn import preprocessing
from matplotlib import pyplot as plt
import math


from sg2vec.data.dataset import SceneGraphDataset
from sg2vec.data.dataset import RawImageDataset
from sg2vec.scene_graph.relation_extractor import Relations
from argparse import ArgumentParser
from tqdm import tqdm

from sg2vec.learning.model.cnn_lstm import CNN_LSTM_Classifier
from sg2vec.learning.model.lstm import LSTM_Classifier
from sg2vec.learning.model.mrgcn import MRGCN
from sg2vec.learning.model.mrgin import MRGIN
from sg2vec.learning.model.cnn import CNN_Classifier
from sg2vec.learning.model.resnet50_lstm import ResNet50_LSTM_Classifier
from torch_geometric.data import Data, DataLoader, DataListLoader
from sklearn.utils.class_weight import compute_class_weight
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
from sklearn.utils import resample
import pickle as pkl
from sklearn.model_selection import train_test_split, StratifiedKFold
from sg2vec.learning.util.metrics import *

from collections import Counter,defaultdict
import wandb

from sg2vec.learning.util.model_input_preprocessing import *

import sg2vec
sys.modules['data'] = sg2vec.data

class Trainer:

    def __init__(self, config, wandb_a = None):
        self.config = config
        if wandb_a != None:
            self.log = True
            self.wandb = wandb_a
            self.wandb_config = wandb_a.config
            # load config into wandb
            for section in self.config.args:
                for config_arg in self.config.args[section]:
                    self.wandb_config[section+'.'+config_arg] = self.config.args[section][config_arg]
        else:
            self.log = False             
        if self.config.training_configuration["seed"] != None: 
            self.config.seed = self.config.training_configuration["seed"]
            np.random.seed(self.config.seed)
            torch.manual_seed(self.config.seed)
        else:
            self.config.seed = random.randint(0,2**32) #default value
            np.random.seed(self.config.seed)
            torch.manual_seed(self.config.seed)
            
        self.feature_list = set()
        for i in range(self.config.training_configuration['num_of_classes']):
            self.feature_list.add("type_"+str(i))
        self.toGPU = lambda x, dtype: torch.as_tensor(x, dtype=dtype, device=self.config.training_configuration['device'])
        self.best_val_loss = 99999
        self.best_epoch = 0
        self.best_val_acc = 0
        self.best_val_auc = 0
        self.best_val_confusion = []
        self.best_val_f1 = 0
        self.best_val_mcc = -1.0
        self.best_val_acc_balanced = 0
        self.unique_clips = {}
        self.log = False

    #abstract function implemented by subclasses
    def split_dataset(self):
        raise NotImplementedError


    def build_model(self): #this involves changing mrcgn and mrgin files to be compatible with new config tht we pass in
        # BD mode
        self.config.num_features = len(self.feature_list)
        self.config.num_relations = max([r.value for r in Relations])+1
        if self.config.model_configuration["model"] == "mrgcn":
            self.model = MRGCN(self.config).to(self.config.training_configuration["device"])
        elif self.config.model_configuration["model"]  == "mrgin":
            self.model = MRGIN(self.config).to(self.config.training_configuration["device"])
        elif self.config.model_configuration["model"]  == "cnn":
            self.model = CNN_Classifier((self.config.training_configuration['batch_size'], self.image_dataset.frame_limit,self.image_dataset.color_channels, self.image_dataset.im_height, self.image_dataset.im_width), self.config).to(self.config.training_configuration["device"])
        elif self.config.model_configuration["model"]  == "cnn_lstm":
            self.model = CNN_LSTM_Classifier((self.config.training_configuration['batch_size'], self.image_dataset.frame_limit,self.image_dataset.color_channels, self.image_dataset.im_height, self.image_dataset.im_width), self.config).to(self.config.training_configuration["device"])
        elif self.config.model_configuration["model"]  == "lstm":
            self.model = LSTM_Classifier((self.config.training_configuration['batch_size'], self.image_dataset.frame_limit,self.image_dataset.color_channels, self.image_dataset.im_height, self.image_dataset.im_width),'lstm', self.config).to(self.config.training_configuration["device"])        
        elif self.config.model_configuration["model"]  == "gru":
            self.model = LSTM_Classifier((self.config.training_configuration['batch_size'], self.image_dataset.frame_limit,self.image_dataset.color_channels, self.image_dataset.im_height, self.image_dataset.im_width), 'gru', self.config).to(self.config.training_configuration["device"]) 
        elif self.config.model_configuration["model"] == "resnet50_lstm":
            self.model = ResNet50_LSTM_Classifier((self.config.training_configuration['batch_size'], self.image_dataset.frame_limit,self.image_dataset.color_channels, self.image_dataset.im_height, self.image_dataset.im_width), self.config).to(self.config.training_configuration["device"]) 
        else:
            raise Exception("model selection is invalid: " + self.config.model_configuration["model"])
        
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.config.training_configuration["learning_rate"], weight_decay=self.config.training_configuration["weight_decay"])
        
        if self.config.model_configuration["load_model"]  == False:
            if self.class_weights.shape[0] < 2:
                self.loss_func = nn.CrossEntropyLoss()
            else:
                self.loss_func = nn.CrossEntropyLoss(weight=self.class_weights.float().to(self.config.training_configuration["device"]))
     
            #wandb.watch(self.model, log="all")
            if self.log:
                self.wandb.watch(self.model, log="all")


    # Pick between Standard Training and KFold Cross Validation Training
    def learn(self):  
        if self.config.training_configuration["n_fold"] <= 1 or self.config.location_data["transfer_path"] != None:
            print('\nRunning Standard Training Loop\n')
            self.train()
        else:
            print('\nRunning {}-Fold Cross Validation Training Loop\n'.format(self.config.training_configuration["n_fold"]))
            self.cross_valid()

    def train(self): #edit
        if (self.config.training_configuration['task_type'] in ['sequence_classification','graph_classification','collision_prediction']):
            tqdm_bar = tqdm(range(self.config.training_configuration['epochs']))
    
            for epoch_idx in tqdm_bar: # iterate through epoch   
                acc_loss_train = 0
                self.sequence_loader = DataListLoader(self.training_data, batch_size=self.config.training_configuration["batch_size"])
    
                for data_list in self.sequence_loader: # iterate through batches of the dataset
                    self.model.train()
                    self.optimizer.zero_grad()
                    labels = torch.empty(0).long().to(self.config.training_configuration["device"])
                    outputs = torch.empty(0,2).to(self.config.training_configuration["device"])
    
                    #need to change below for current implementation
                    for sequence in data_list: # iterate through scene-graph sequences in the batch
                        data, label = sequence['sequence'], sequence['label'] 
                        graph_list = [Data(x=g['node_features'], edge_index=g['edge_index'], edge_attr=g['edge_attr']) for g in data]  
                        self.train_loader = DataLoader(graph_list, batch_size=len(graph_list))
                        sequence = next(iter(self.train_loader)).to(self.config.training_configuration["device"])
                        output, _ = self.model.forward(sequence.x, sequence.edge_index, sequence.edge_attr, sequence.batch)
                        if self.config.training_configuration['task_type'] == 'sequence_classification': # seq vs graph based learning
                            labels  = torch.cat([labels, torch.LongTensor([label]).to(self.config.training_configuration["device"])], dim=0)
                        elif self.config.training_configuration['task_type'] in ['collision_prediction']:
                            label = torch.LongTensor(np.full(output.shape[0], label)).to(self.config.training_configuration["device"]) #fill label to length of the sequence. shape (len_input_sequence, 1)
                            labels  = torch.cat([labels, label], dim=0)
                        else:
                            raise ValueError('task_type is unimplemented')
                        outputs = torch.cat([outputs, output.view(-1, 2)], dim=0) #in this case the output is of shape (len_input_sequence, 2)
                    loss_train = self.loss_func(outputs, labels)
                    loss_train.backward()
                    acc_loss_train += loss_train.detach().cpu().item() * len(data_list)
                    self.optimizer.step()
                    del loss_train
    
                acc_loss_train /= len(self.training_data)
                tqdm_bar.set_description('Epoch: {:04d}, loss_train: {:.4f}'.format(epoch_idx, acc_loss_train))
    
                if epoch_idx % self.config.training_configuration["test_step"] == 0:
                    self.evaluate(epoch_idx)
                    
        else:
            raise ValueError('train(): task type error') 

    
    def save_model(self):
        """Function to save the model."""
        saved_path = Path(self.config.training_configuration["model_save_path"]).resolve()
        os.makedirs(os.path.dirname(saved_path), exist_ok=True)
        torch.save(self.model.state_dict(), str(saved_path))
        with open(os.path.dirname(saved_path) + "/model_parameters.txt", "w+") as f:
            f.write(str(self.config))
            f.write('\n')
            f.write(str(' '.join(sys.argv)))

    def load_model(self):
        """Function to load the model."""
        saved_path = Path(self.config.training_configuration["model_load_path"]).resolve()
        if saved_path.exists():
            self.build_model()
            self.model.load_state_dict(torch.load(str(saved_path)))
            self.model.eval()
