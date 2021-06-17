import os, sys, pdb
sys.path.append(os.path.dirname(sys.path[0]))
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


from data.dataset import SceneGraphDataset
from data.dataset import RawImageDataset
from scene_graph.relation_extractor import Relations
from argparse import ArgumentParser
from pathlib import Path
from tqdm import tqdm
from learning.model.mrgcn import MRGCN
from learning.model.mrgin import MRGIN
from learning.model.cnn import CNN_Classifier
from torch_geometric.data import Data, DataLoader, DataListLoader
from sklearn.utils.class_weight import compute_class_weight
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
from sklearn.utils import resample
import pickle as pkl
from sklearn.model_selection import train_test_split, StratifiedKFold
from learning.util.metrics import *

from collections import Counter,defaultdict
import wandb

from learning.util.model_input_preprocessing import *

class Trainer:

    def __init__(self, config, wandb_a = None):
        self.config = config
        self.wandb = wandb_a
        self.wandb_config = wandb_a.config
        if self.config.training_configuration["seed"] != None: 
            self.config.seed = self.config.training_configuration["seed"]
            np.random.seed(self.config.seed)
            torch.manual_seed(self.config.seed)
        else:
            self.config.seed = random.randint(0,2**32) #default value
            np.random.seed(self.config.seed)
            torch.manual_seed(self.config.seed)
            
        # load config into wandb
        for section in self.config.args:
            for config_arg in self.config.args[section]:
                self.wandb_config[section+'.'+config_arg] = self.config.args[section][config_arg]
                  
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


    def split_dataset(self): #this is init_dataset from multimodal
        if self.config.training_configuration['task_type'] == 'cnn image classification':
            self.training_data, self.testing_data, self.feature_list = self.build_real_image_dataset(self.config.location_data["input_path"], self.config.training_configuration["split_ratio"], downsample=self.config.training_configuration["downsample"], seed=self.config.seed, transfer_path=self.config.location_data["transfer_path"])
            self.training_labels = np.array([ i[1] for i in self.training_data])
            self.testing_labels = np.array([ i[1] for i in self.testing_data])
            self.training_clip_name = np.array([ i[2] for i in self.training_data])
            self.testing_clip_name = np.array([ i[2] for i in self.testing_data])
            self.training_data = np.array([i[0].numpy() for i in self.training_data])
            self.testing_data = np.array([i[0].numpy() for i in self.testing_data])
            self.class_weights = torch.from_numpy(compute_class_weight('balanced', np.unique(self.training_labels), self.training_labels))
            
            if self.config.training_configuration["n_fold"] <= 1:
                print("Number of Training Sequences Included: ", len(self.training_data))
                print("Number of Testing Sequences Included: ", len(self.testing_data))
                print("Num of Training Labels in Each Class: " + str(np.unique(self.training_labels, return_counts=True)[1]) + ", Class Weights: " + str(self.class_weights))
                print("Num of Testing Labels in Each Class: " + str(np.unique(self.testing_labels, return_counts=True)[1]) + ", Class Weights: " + str(self.class_weights)) 
     
        elif (self.config.training_configuration['task_type'] in ['sequence_classification','graph_classification','collision_prediction']):
            self.training_data, self.testing_data, self.feature_list = self.build_scenegraph_dataset(self.config.location_data["input_path"], self.config.training_configuration["split_ratio"], downsample=self.config.training_configuration["downsample"], seed=self.config.seed, transfer_path=self.config.location_data["transfer_path"])
            self.total_train_labels = np.concatenate([np.full(len(data['sequence']), data['label']) for data in self.training_data]) # used to compute frame-level class weighting
            self.total_test_labels  = np.concatenate([np.full(len(data['sequence']), data['label']) for data in self.testing_data])
            self.training_labels = [data['label'] for data in self.training_data]
            self.testing_labels  = [data['label'] for data in self.testing_data]
            if self.config.training_configuration['task_type'] == 'sequence_classification':
                self.class_weights = torch.from_numpy(compute_class_weight('balanced', np.unique(self.training_labels), self.training_labels))
                if self.config.training_configuration["n_fold"] <= 1:
                    print("Number of Sequences Included: ", len(self.training_data))
                    print("Num Labels in Each Class: " + str(np.unique(self.training_labels, return_counts=True)[1]) + ", Class Weights: " + str(self.class_weights))
            elif self.config.training_configuration['task_type'] == 'collision_prediction':
                self.class_weights = torch.from_numpy(compute_class_weight('balanced', np.unique(self.total_train_labels), self.total_train_labels))
                if self.config.training_configuration["n_fold"] <= 1:
                    print("Number of Training Sequences Included: ", len(self.training_data))
                    print("Number of Testing Sequences Included: ", len(self.testing_data))
                    print("Number of Training Labels in Each Class: " + str(np.unique(self.total_train_labels, return_counts=True)[1]) + ", Class Weights: " + str(self.class_weights))
                    print("Number of Testing Labels in Each Class: " + str(np.unique(self.total_test_labels, return_counts=True)[1]) + ", Class Weights: " + str(self.class_weights))
        else:
            raise ValueError('split_dataset(): task type error') 


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
        else:
            raise Exception("model selection is invalid: " + self.config.model_configuration["model"])
        #
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.config.training_configuration["learning_rate"], weight_decay=self.config.training_configuration["weight_decay"])
        
        
        if self.class_weights.shape[0] < 2:
            self.loss_func = nn.CrossEntropyLoss()
        else:
            self.loss_func = nn.CrossEntropyLoss(weight=self.class_weights.float().to(self.config.training_configuration["device"]))

        #wandb.watch(self.model, log="all")
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

        elif self.config.training_configuration['task_type'] == 'cnn image classification':
            tqdm_bar = tqdm(range(self.config.training_configuration['epochs']))
            for epoch_idx in tqdm_bar: # iterate through epoch   
                acc_loss_train = 0
                permutation = np.random.permutation(len(self.training_data)) # shuffle dataset before each epoch
                self.model.train()
                for i in range(0, len(self.training_data), self.config.training_configuration['batch_size']): # iterate through batches of the dataset
                    batch_index = i + self.config.training_configuration['batch_size'] if i + self.config.training_configuration['batch_size'] <= len(self.training_data) else len(self.training_data)
                    indices = permutation[i:batch_index]
                    batch_x, batch_y = self.training_data[indices], self.training_labels[indices]
                    batch_x, batch_y = self.toGPU(batch_x, torch.float32), self.toGPU(batch_y, torch.long)
                    output = self.model.forward(batch_x).view(-1, 2)
                    loss_train = self.loss_func(output, batch_y)
                    loss_train.backward()
                    acc_loss_train += loss_train.detach().cpu().item() * len(indices)
                    self.optimizer.step()
                    del loss_train
    
                acc_loss_train /= len(self.training_data)
                tqdm_bar.set_description('Epoch: {:04d}, loss_train: {:.4f}'.format(epoch_idx, acc_loss_train))
                
                # no cross validation 
                if epoch_idx % self.config.training_configuration['test_step'] == 0:
                    self.eval_model(epoch_idx)
                    
        else:
            raise ValueError('train(): task type error') 
####

    
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
            
            
            
class Scenegraph_Trainer(Trainer):
    #optimize call. build_scenegraph_dataset was changed to self
    def build_scenegraph_dataset(self,cache_path, train_to_test_ratio=0.3, downsample=False, seed=0, transfer_path=None): #this creates test and training datasets
        '''
        Dataset format 
            scenegraphs_sequence: dict_keys(['sequence', 'label', 'folder_name'])
                'sequence': scenegraph metadata
                'label': classification output [0 -> non_risky (negative), 1 -> risky (positive)]
                'folder_name': foldername storing sequence data
    
        Dataset modes
            no downsample
                all sequences used for the train and test set regardless of class distribution
            downsample  
                equal amount of positive and negative sequences used for the train and test set
            transfer 
                replaces original test set with another dataset 
        '''
    #     dataset_file = open(cache_path, "rb")
    #     scenegraphs_sequence, feature_list = pkl.load(dataset_file)
    
        #Load sg dataset obj
        scene_graph_dataset  = SceneGraphDataset()
        scene_graph_dataset.dataset_save_path = self.config.location_data["input_path"]
        self.scene_graph_dataset = scene_graph_dataset.load()
        #scenegraphs_sequence = self.scene_graph_dataset.scene_graphs
        self.feature_list = set() #is this right or shld it be done sooner?, not really used in current sg extraction pipeline
        for i in range(self.config.training_configuration["num_of_classes"]): #    if so need to add num_classes to config
            self.feature_list.add("type_"+str(i))
            
    
        class_0 = []
        class_1 = []
        sorted_seq = sorted(self.scene_graph_dataset.labels)
        if self.config.training_configuration["scenegraph_dataset_type"] == "carla":
            for ind, seq in enumerate(sorted_seq): #for each seq in labels
                data_to_append = {"sequence":process_carla_graph_sequences(self.scene_graph_dataset.scene_graphs[seq], self.feature_list, folder_name = self.scene_graph_dataset.folder_names[ind] ), "label":self.scene_graph_dataset.labels[seq], "folder_name": self.scene_graph_dataset.folder_names[ind]}
                if self.scene_graph_dataset.labels[seq] == 0:
#                     class_0.append(scene_graph_dataset.scene_graphs[seq]) #right now we are appending whole dictionary that contains data for all frame sg, shld we instead append each frame's sg separately
                    class_0.append(data_to_append)  #maybe individually for graph based and all frames together in one for seq based?
                                                                        #maybe instead create a massive dict with the form {seq:scene_graph_dataset.scene_graphs[seq], scene_graph_dataset.labels[seq]...}
                elif self.scene_graph_dataset.labels[seq] == 1:
                    class_1.append(data_to_append)
                    
        elif self.config.training_configuration["scenegraph_dataset_type"] == "real":
            for ind, seq in enumerate(sorted_seq): 
                data_to_append = {"sequence":process_real_image_graph_sequences(self.scene_graph_dataset.scene_graphs[seq], self.feature_list, folder_name = self.scene_graph_dataset.folder_names[ind] ), "label":self.scene_graph_dataset.labels[seq], "folder_name": self.scene_graph_dataset.folder_names[ind]}
                if self.scene_graph_dataset.labels[seq] == 0:
#                     class_0.append(scene_graph_dataset.scene_graphs[seq]) #right now we are appending whole dictionary that contains data for all frame sg, shld we instead append each frame's sg separately
                    class_0.append(data_to_append)  #maybe individually for graph based and all frames together in one for seq based?
                                                                        #maybe instead create a massive dict with the form {seq:scene_graph_dataset.scene_graphs[seq], scene_graph_dataset.labels[seq]...}
                elif self.scene_graph_dataset.labels[seq] == 1:
                    class_1.append(data_to_append)
        #else
        
            
        y_0 = [0]*len(class_0)  
        y_1 = [1]*len(class_1)

        min_number = min(len(class_0), len(class_1))
        
        if downsample:
            modified_class_0, modified_y_0 = resample(class_0, y_0, n_samples=min_number)
        else:
            modified_class_0, modified_y_0 = class_0, y_0
        train, test, train_y, test_y = train_test_split(modified_class_0+class_1, modified_y_0+y_1, test_size=train_to_test_ratio, shuffle=True, stratify=modified_y_0+y_1, random_state=seed)
        if self.config.location_data["transfer_path"] != None:#what is this meant to do if input path is meant to load in sq dataset obj?
            test, _ = pkl.load(open(self.config.location_data["transfer_path"], "rb"))
            scenegraphs_sequence = class_1+class_0
            return scenegraphs_sequence, test, self.feature_list 
        #dont do kfold here instead it is done when learn() is called
        return train, test, self.feature_list # redundant return of self.feature_list
    

    def format_use_case_model_input(self, sequence):
        #####################
        ''' move to another file'''
        if self.config.training_configuration["scenegraph_dataset_type"] == "carla":
            for seq in sequence.scene_graph_dataset.scene_graphs:
                data = {"sequence":process_carla_graph_sequences(sequence.scene_graph_dataset.scene_graphs[seq], feature_list = None, folder_name = sequence.scene_graph_dataset.folder_names[seq]) , "label":None, "folder_name": sequence.scene_graph_dataset.folder_names[seq]}
        elif self.config.training_configuration["scenegraph_dataset_type"] == "real":
            for seq in sequence.scene_graph_dataset.scene_graphs:
                data = {"sequence":process_real_image_graph_sequences(sequence.scene_graph_dataset.scene_graphs[seq], feature_list = None, folder_name = sequence.scene_graph_dataset.folder_names[seq]) , "label":None, "folder_name": sequence.scene_graph_dataset.folder_names[seq]}
        else:
            raise ValueError('output():scenegraph_dataset_type unrecognized')
        #####################
        data = data['sequence']
        graph_list = [Data(x=g['node_features'], edge_index=g['edge_index'], edge_attr=g['edge_attr']) for g in data]  
        train_loader = DataLoader(graph_list, batch_size=len(graph_list))
        sequence = next(iter(train_loader)).to(self.config.training_configuration["device"])
        
        return (sequence.x, sequence.edge_index, sequence.edge_attr, sequence.batch)    
    
                
    def cross_valid(self): #edit
    
           # KFold cross validation with similar class distribution in each fold
           skf = StratifiedKFold(n_splits=self.config.training_configuration["n_fold"])
           X = np.array(self.training_data + self.testing_data)
           y = np.array(self.training_labels + self.testing_labels)
    
           # self.results stores average metrics for the the n_folds
           self.results = {}
           self.fold = 1
    
           # Split training and testing data based on n_splits (Folds)
           for train_index, test_index in skf.split(X, y):
               X_train, X_test = X[train_index], X[test_index]
               y_train, y_test = y[train_index], y[test_index]
    
               self.training_data = X_train
               self.testing_data  = X_test
               self.training_labels = y_train
               self.testing_labels  = y_test
    
               # To compute frame-level class weighting
    #             total_train_labels = np.concatenate([np.full(len(data['sequence']), data['label']) for data in self.training_data]) 
    #             total_test_labels = np.concatenate([np.full(len(data['sequence']), data['label']) for data in self.testing_data])
    #             self.class_weights = torch.from_numpy(compute_class_weight('balanced', np.unique(self.total_train_labels), self.total_train_labels))
               #^^^ repetitive, we alrdy calculate this in split_dataset()
               if self.config.training_configuration['task_type'] == 'sequence_classification':
                   print('\nFold {}'.format(self.fold))
                   print("Number of Sequences Included: ", len(self.training_data))
                   print("Num Labels in Each Class: " + str(np.unique(self.training_labels, return_counts=True)[1]) + ", Class Weights: " + str(self.class_weights))
      
               elif self.config.training_configuration['task_type'] == 'collision_prediction':
                   print('\nFold {}'.format(self.fold))
                   print("Number of Training Sequences Included: ", len(self.training_data))
                   print("Number of Testing Sequences Included: ",  len(self.testing_data))
                   print("Number of Training Labels in Each Class: " + str(np.unique(self.total_train_labels, return_counts=True)[1]) + ", Class Weights: " + str(self.class_weights))
                   print("Number of Testing Labels in Each Class: " + str(np.unique(self.total_test_labels, return_counts=True)[1]) + ", Class Weights: " + str(self.class_weights))
               
               
               
               self.best_val_loss = 99999
               self.train()
               self.log = True
               outputs_train, labels_train, outputs_test, labels_test, metrics = self.evaluate(self.fold)
               self.update_sg_cross_valid_metrics(outputs_train, labels_train, outputs_test, labels_test, metrics)
               self.log = False
    
               if self.fold != self.config.training_configuration["n_fold"]:            
                   del self.model
                   del self.optimizer
                   self.build_model()
                   
               self.fold += 1            
           del self.results
                   
                   
    def inference(self, testing_data, testing_labels):
            labels = torch.LongTensor().to(self.config.training_configuration["device"])
            outputs = torch.FloatTensor().to(self.config.training_configuration["device"])
            acc_loss_test = 0
            attns_weights = []
            node_attns = []
            sum_prediction_frame = 0
            sum_seq_len = 0
            num_risky_sequences = 0
            num_safe_sequences = 0
            sum_predicted_risky_indices = 0 #sum is calculated as (value * (index+1))/sum(range(seq_len)) for each value and index in the sequence.
            sum_predicted_safe_indices = 0  #sum is calculated as ((1-value) * (index+1))/sum(range(seq_len)) for each value and index in the sequence.
            inference_time = 0
            prof_result = ""
            correct_risky_seq = 0
            correct_safe_seq = 0
            incorrect_risky_seq = 0
            incorrect_safe_seq = 0
    
            with torch.autograd.profiler.profile(enabled=False, use_cuda=True) as prof:
                with torch.no_grad():
                    for i in range(len(testing_data)): # iterate through scenegraphs
                        data, label = testing_data[i]['sequence'], testing_labels[i]
                        data_list = [Data(x=g['node_features'], edge_index=g['edge_index'], edge_attr=g['edge_attr']) for g in data]
                        self.test_loader = DataLoader(data_list, batch_size=len(data_list))
                        sequence = next(iter(self.test_loader)).to(self.config.training_configuration["device"])
                        self.model.eval()
                        #start = torch.cuda.Event(enable_timing=True)
                        #end =  torch.cuda.Event(enable_timing=True)
                        #start.record()
                        output, attns = self.model.forward(sequence.x, sequence.edge_index, sequence.edge_attr, sequence.batch)
                        #end.record()
                        #torch.cuda.synchronize()
                        inference_time += 0#start.elapsed_time(end)
                        output = output.view(-1,2)
                        label = torch.LongTensor(np.full(output.shape[0], label)).to(self.config.training_configuration["device"]) #fill label to length of the sequence.
    
                        #log metrics for risky and non-risky clips separately.
                        if(1 in label):
                            preds = output.max(1)[1].type_as(label)
                            num_risky_sequences += 1
                            sum_seq_len += output.shape[0]
                            if (1 in preds):
                                correct_risky_seq += 1 #sequence level metrics
                                sum_prediction_frame += torch.where(preds == 1)[0][0].item() #returns the first index of a "risky" prediction in this sequence.
                                sum_predicted_risky_indices += torch.sum(torch.where(preds==1)[0]+1).item()/np.sum(range(output.shape[0]+1)) #(1*index)/seq_len added to sum.
                            else:
                                incorrect_risky_seq += 1
                                sum_prediction_frame += output.shape[0] #if no risky predictions are made, then add the full sequence length to running avg.
                        elif(0 in label):
                            preds = output.max(1)[1].type_as(label)
                            num_safe_sequences += 1
                            if (1 in preds): #sequence level metrics
                                incorrect_safe_seq += 1 
                            else:
                                correct_safe_seq += 1 
    
                            if (0 in preds):
                                sum_predicted_safe_indices += torch.sum(torch.where(preds==0)[0]+1).item()/np.sum(range(output.shape[0]+1)) #(1*index)/seq_len added to sum.
    
                        loss_test = self.loss_func(output, label)
                        acc_loss_test += loss_test.detach().cpu().item()
    
                        outputs = torch.cat([outputs, output], dim=0)
                        labels = torch.cat([labels, label], dim=0)
    
                        # if 'lstm_attn_weights' in attns:
                        #     attns_weights.append(attns['lstm_attn_weights'].squeeze().detach().cpu().numpy().tolist())
                        # if 'pool_score' in attns:
                        #     node_attn = {}
                        #     node_attn["original_batch"] = sequence.batch.detach().cpu().numpy().tolist()
                        #     node_attn["pool_perm"] = attns['pool_perm'].detach().cpu().numpy().tolist()
                        #     node_attn["pool_batch"] = attns['batch'].detach().cpu().numpy().tolist()
                        #     node_attn["pool_score"] = attns['pool_score'].detach().cpu().numpy().tolist()
                        #     node_attns.append(node_attn)
    
            avg_risky_prediction_frame = sum_prediction_frame / num_risky_sequences #avg of first indices in a sequence that a risky frame is first correctly predicted.
            avg_risky_seq_len = sum_seq_len / num_risky_sequences #sequence length for comparison with the prediction frame metric. 
            avg_predicted_risky_indices = sum_predicted_risky_indices / num_risky_sequences
            avg_predicted_safe_indices = sum_predicted_safe_indices / num_safe_sequences
            seq_tpr = correct_risky_seq / num_risky_sequences
            seq_fpr = incorrect_safe_seq / num_safe_sequences
            seq_tnr = correct_safe_seq / num_safe_sequences
            seq_fnr = incorrect_risky_seq / num_risky_sequences
            if prof != None:
                prof_result = prof.key_averages().table(sort_by="cuda_time_total")
    
            return outputs, \
                    labels, \
                    acc_loss_test/len(testing_data), \
                    attns_weights, \
                    node_attns, \
                    avg_risky_prediction_frame, \
                    avg_risky_seq_len, \
                    avg_predicted_risky_indices, \
                    avg_predicted_safe_indices, \
                    inference_time, \
                    prof_result, \
                    seq_tpr, \
                    seq_fpr, \
                    seq_tnr, \
                    seq_fnr

   


    def evaluate(self, current_epoch=None):
        metrics = {}
        outputs_train, \
        labels_train, \
        acc_loss_train, \
        attns_train, \
        node_attns_train, \
        train_avg_prediction_frame, \
        train_avg_seq_len, \
        avg_predicted_risky_indices, \
        avg_predicted_safe_indices, \
        train_inference_time, \
        train_profiler_result, \
        seq_tpr, \
        seq_fpr, \
        seq_tnr, \
        seq_fnr = self.inference(self.training_data, self.training_labels)

        metrics['train'] = get_metrics(outputs_train, labels_train)
        metrics['train']['loss'] = acc_loss_train
        metrics['train']['avg_prediction_frame'] = train_avg_prediction_frame
        metrics['train']['avg_seq_len'] = train_avg_seq_len
        metrics['train']['avg_predicted_risky_indices'] = avg_predicted_risky_indices
        metrics['train']['avg_predicted_safe_indices'] = avg_predicted_safe_indices
        metrics['train']['seq_tpr'] = seq_tpr
        metrics['train']['seq_tnr'] = seq_tnr
        metrics['train']['seq_fpr'] = seq_fpr
        metrics['train']['seq_fnr'] = seq_fnr
        with open("graph_profile_metrics.txt", mode='w') as f:
            f.write(train_profiler_result)

        outputs_test, \
        labels_test, \
        acc_loss_test, \
        attns_test, \
        node_attns_test, \
        val_avg_prediction_frame, \
        val_avg_seq_len, \
        avg_predicted_risky_indices, \
        avg_predicted_safe_indices, \
        test_inference_time, \
        test_profiler_result, \
        seq_tpr, \
        seq_fpr, \
        seq_tnr, \
        seq_fnr = self.inference(self.testing_data, self.testing_labels)

        metrics['test'] = get_metrics(outputs_test, labels_test)
        metrics['test']['loss'] = acc_loss_test
        metrics['test']['avg_prediction_frame'] = val_avg_prediction_frame
        metrics['test']['avg_seq_len'] = val_avg_seq_len
        metrics['test']['avg_predicted_risky_indices'] = avg_predicted_risky_indices
        metrics['test']['avg_predicted_safe_indices'] = avg_predicted_safe_indices
        metrics['test']['seq_tpr'] = seq_tpr
        metrics['test']['seq_tnr'] = seq_tnr
        metrics['test']['seq_fpr'] = seq_fpr
        metrics['test']['seq_fnr'] = seq_fnr
        metrics['avg_inf_time'] = (train_inference_time + test_inference_time) / (len(labels_train) + len(labels_test))

        print("\ntrain loss: " + str(acc_loss_train) + ", acc:", metrics['train']['acc'], metrics['train']['confusion'], "mcc:", metrics['train']['mcc'], \
              "\ntest loss: " +  str(acc_loss_test) + ", acc:",  metrics['test']['acc'],  metrics['test']['confusion'], "mcc:", metrics['test']['mcc'])
        
        self.update_sg_best_metrics(metrics, current_epoch)
        metrics['best_epoch'] = self.best_epoch
        metrics['best_val_loss'] = self.best_val_loss
        metrics['best_val_acc'] = self.best_val_acc
        metrics['best_val_auc'] = self.best_val_auc
        metrics['best_val_conf'] = self.best_val_confusion
        metrics['best_val_f1'] = self.best_val_f1
        metrics['best_val_mcc'] = self.best_val_mcc
        metrics['best_val_acc_balanced'] = self.best_val_acc_balanced
        metrics['best_avg_pred_frame'] = self.best_avg_pred_frame
        
        if self.config.training_configuration["n_fold"] <= 1 or self.log:
            log_wandb(metrics)
        
        return outputs_train, labels_train, outputs_test, labels_test, metrics


        #automatically save the model and metrics with the lowest validation loss
    def update_sg_best_metrics(self, metrics, current_epoch):
        if metrics['test']['loss'] < self.best_val_loss:
            self.best_val_loss = metrics['test']['loss']
            self.best_epoch = current_epoch if current_epoch != None else self.config.training_configuration["epochs"]
            self.best_val_acc = metrics['test']['acc']
            self.best_val_auc = metrics['test']['auc']
            self.best_val_confusion = metrics['test']['confusion']
            self.best_val_f1 = metrics['test']['f1']
            self.best_val_mcc = metrics['test']['mcc']
            self.best_val_acc_balanced = metrics['test']['balanced_acc']
            self.best_avg_pred_frame = metrics['test']['avg_prediction_frame']
            #self.save_model()

    
    # Averages metrics after the end of each cross validation fold
    def update_sg_cross_valid_metrics(self, outputs_train, labels_train, outputs_test, labels_test, metrics):
        if self.fold == 1:
            self.results['outputs_train'] = outputs_train
            self.results['labels_train'] = labels_train
            self.results['train'] = metrics['train']
            self.results['train']['loss'] = metrics['train']['loss']
            self.results['train']['avg_prediction_frame'] = metrics['train']['avg_prediction_frame'] 
            self.results['train']['avg_seq_len']  = metrics['train']['avg_seq_len'] 
            self.results['train']['avg_predicted_risky_indices'] = metrics['train']['avg_predicted_risky_indices'] 
            self.results['train']['avg_predicted_safe_indices'] = metrics['train']['avg_predicted_safe_indices']

            self.results['outputs_test'] = outputs_test
            self.results['labels_test'] = labels_test
            self.results['test'] = metrics['test']
            self.results['test']['loss'] = metrics['test']['loss'] 
            self.results['test']['avg_prediction_frame'] = metrics['test']['avg_prediction_frame'] 
            self.results['test']['avg_seq_len'] = metrics['test']['avg_seq_len'] 
            self.results['test']['avg_predicted_risky_indices'] = metrics['test']['avg_predicted_risky_indices'] 
            self.results['test']['avg_predicted_safe_indices'] = metrics['test']['avg_predicted_safe_indices']
            self.results['avg_inf_time'] = metrics['avg_inf_time']

            self.results['best_epoch']    = metrics['best_epoch']
            self.results['best_val_loss'] = metrics['best_val_loss']
            self.results['best_val_acc']  = metrics['best_val_acc']
            self.results['best_val_auc']  = metrics['best_val_auc']
            self.results['best_val_conf'] = metrics['best_val_conf']
            self.results['best_val_f1']   = metrics['best_val_f1']
            self.results['best_val_mcc']  = metrics['best_val_mcc']
            self.results['best_val_acc_balanced'] = metrics['best_val_acc_balanced']
            self.results['best_avg_pred_frame'] = metrics['best_avg_pred_frame']
        else:
            self.results['outputs_train'] = torch.cat((self.results['outputs_train'], outputs_train), dim=0)
            self.results['labels_train']  = torch.cat((self.results['labels_train'], labels_train), dim=0)
            self.results['train']['loss'] = np.append(self.results['train']['loss'], metrics['train']['loss'])
            self.results['train']['avg_prediction_frame'] = np.append(self.results['train']['avg_prediction_frame'], 
                                                                            metrics['train']['avg_prediction_frame'])
            self.results['train']['avg_seq_len']  = np.append(self.results['train']['avg_seq_len'], metrics['train']['avg_seq_len'])
            self.results['train']['avg_predicted_risky_indices'] = np.append(self.results['train']['avg_predicted_risky_indices'], 
                                                                                    metrics['train']['avg_predicted_risky_indices'])
            self.results['train']['avg_predicted_safe_indices'] = np.append(self.results['train']['avg_predicted_safe_indices'], 
                                                                                    metrics['train']['avg_predicted_safe_indices'])
            
            self.results['outputs_test'] = torch.cat((self.results['outputs_test'], outputs_test), dim=0)
            self.results['labels_test']  = torch.cat((self.results['labels_test'], labels_test), dim=0)
            self.results['test']['loss'] = np.append(self.results['test']['loss'], metrics['test']['loss'])
            self.results['test']['avg_prediction_frame'] = np.append(self.results['test']['avg_prediction_frame'], 
                                                                        metrics['test']['avg_prediction_frame'])
            self.results['test']['avg_seq_len'] = np.append(self.results['test']['avg_seq_len'], metrics['test']['avg_seq_len'])
            self.results['test']['avg_predicted_risky_indices'] = np.append(self.results['test']['avg_predicted_risky_indices'], 
                                                                                    metrics['test']['avg_predicted_risky_indices'])
            self.results['test']['avg_predicted_safe_indices'] = np.append(self.results['test']['avg_predicted_safe_indices'], 
                                                                                    metrics['test']['avg_predicted_safe_indices'])
            self.results['avg_inf_time'] = np.append(self.results['avg_inf_time'], metrics['avg_inf_time'])

            self.results['best_epoch']    = np.append(self.results['best_epoch'], metrics['best_epoch'])
            self.results['best_val_loss'] = np.append(self.results['best_val_loss'], metrics['best_val_loss'])
            self.results['best_val_acc']  = np.append(self.results['best_val_acc'], metrics['best_val_acc'])
            self.results['best_val_auc']  = np.append(self.results['best_val_auc'], metrics['best_val_auc'])
            self.results['best_val_conf'] = np.append(self.results['best_val_conf'], metrics['best_val_conf'])
            self.results['best_val_f1']   = np.append(self.results['best_val_f1'], metrics['best_val_f1'])
            self.results['best_val_mcc']  = np.append(self.results['best_val_mcc'], metrics['best_val_mcc'])
            self.results['best_val_acc_balanced'] = np.append(self.results['best_val_acc_balanced'], metrics['best_val_acc_balanced'])
            self.results['best_avg_pred_frame'] = np.append(self.results['best_avg_pred_frame'], metrics['best_avg_pred_frame'])
            
        # Log final averaged results
        if self.fold == self.config.training_configuration["n_fold"]:
            final_results = {}
            final_results['train'] = get_metrics(self.results['outputs_train'], self.results['labels_train'])
            final_results['train']['loss'] = np.average(self.results['train']['loss'])
            final_results['train']['avg_prediction_frame'] = np.average(self.results['train']['avg_prediction_frame'])
            final_results['train']['avg_seq_len'] = np.average(self.results['train']['avg_seq_len'])
            final_results['train']['avg_predicted_risky_indices'] = np.average(self.results['train']['avg_predicted_risky_indices'])
            final_results['train']['avg_predicted_safe_indices'] = np.average(self.results['train']['avg_predicted_safe_indices'])
            
            final_results['test'] = get_metrics(self.results['outputs_test'], self.results['labels_test'])
            final_results['test']['loss'] = np.average(self.results['test']['loss'])
            final_results['test']['avg_prediction_frame'] = np.average(self.results['test']['avg_prediction_frame'])
            final_results['test']['avg_seq_len'] = np.average(self.results['test']['avg_seq_len'])
            final_results['test']['avg_predicted_risky_indices'] = np.average(self.results['test']['avg_predicted_risky_indices'])
            final_results['test']['avg_predicted_safe_indices'] = np.average(self.results['test']['avg_predicted_safe_indices'])
            final_results['avg_inf_time'] = np.average(self.results['avg_inf_time'])

            # Best results
            final_results['best_epoch']    = np.average(self.results['best_epoch'])
            final_results['best_val_loss'] = np.average(self.results['best_val_loss'])
            final_results['best_val_acc']  = np.average(self.results['best_val_acc'])
            final_results['best_val_auc']  = np.average(self.results['best_val_auc'])
            final_results['best_val_conf'] = self.results['best_val_conf']
            final_results['best_val_f1']   = np.average(self.results['best_val_f1'])
            final_results['best_val_mcc']  = np.average(self.results['best_val_mcc'])
            final_results['best_val_acc_balanced'] = np.average(self.results['best_val_acc_balanced'])
            final_results['best_avg_pred_frame'] = np.average(self.results['best_avg_pred_frame'])

            print('\nFinal Averaged Results')
            print("\naverage train loss: " + str(final_results['train']['loss']) + ", average acc:", final_results['train']['acc'], final_results['train']['confusion'], final_results['train']['auc'], \
                "\naverage test loss: " +  str(final_results['test']['loss']) + ", average acc:", final_results['test']['acc'],  final_results['test']['confusion'], final_results['test']['auc'])

            log_wandb(final_results)
            
            return self.results['outputs_train'], self.results['labels_train'], self.results['outputs_test'], self.results['labels_test'], final_results
                
                
class Image_Trainer(Trainer):
    def build_real_image_dataset(self,cache_path, train_to_test_ratio=0.3, downsample=False, seed=0, transfer_path=None):
          image_dataset = RawImageDataset()
          image_dataset.dataset_save_path = self.config.location_data["input_path"]
          self.image_dataset = image_dataset.load()
          
          self.feature_list = set()
          for i in range(self.config.training_configuration['num_of_classes']):
              self.feature_list.add("type_"+str(i))
              
          class_0 = []
          class_1 = []
          class_0_clip_name = []
          class_1_clip_name = []
          
          for seq in tqdm(self.image_dataset.labels): # for each seq (num total seq,frame,chan,h,w)
              category = self.image_dataset.action_types[seq]
              if category in self.unique_clips:
                  self.unique_clips[category] += 1
              else:
                  self.unique_clips[category] = 1
              seq_data = np.array(process_cnn_image_data(self.image_dataset.data[seq], self.image_dataset.color_channels, self.image_dataset.im_height, self.image_dataset.im_width))
              seq_data = torch.from_numpy(seq_data)
              if self.image_dataset.labels[seq] == 0:
                  class_0.append((seq_data,0,category))                                                  
              elif self.image_dataset.labels[seq] == 1:
                  class_1.append((seq_data,1,category))
          y_0 = [0]*len(class_0)  
          y_1 = [1]*len(class_1)
    
    
          min_number = min(len(class_0), len(class_1))
          
          if downsample:
              modified_class_0, modified_y_0 = resample(class_0, y_0, n_samples=min_number)
          else:
              modified_class_0, modified_y_0 = class_0, y_0
          train, test, train_y, test_y = train_test_split(modified_class_0+class_1, modified_y_0+y_1, test_size=train_to_test_ratio, shuffle=True, stratify=modified_y_0+y_1, random_state=seed)
          if self.config.location_data["transfer_path"] != None:#what is this meant to do if input path is meant to load in sq dataset obj?
              test, _ = pkl.load(open(self.config.location_data["transfer_path"], "rb"))
              image_sequence = class_1+class_0
              return image_sequence, test, self.feature_list 
          #dont do kfold here instead it is done when learn() is called
          return train, test, self.feature_list # redundant return of self.feature_list                    
              
    def model_inference(self, X, y, clip_name):
          labels = torch.LongTensor().to(self.config.training_configuration['device'])
          outputs = torch.FloatTensor().to(self.config.training_configuration['device'])
          # Dictionary storing (output, label) pair for all driving categories
          categories = dict.fromkeys(self.unique_clips)
          for key, val in categories.items():
              categories[key] = {'outputs': outputs, 'labels': labels}
          batch_size = self.config.training_configuration['batch_size'] # NOTE: set to 1 when profiling or calculating inference time.
          acc_loss = 0
          inference_time = 0
          prof_result = ""
    
          with torch.autograd.profiler.profile(enabled=False, use_cuda=True) as prof:
              with torch.no_grad():
                  self.model.eval()
    
                  for i in range(0, len(X), batch_size): # iterate through subsequences
                      batch_index = i + batch_size if i + batch_size <= len(X) else len(X)
                      batch_x, batch_y, batch_clip_name = X[i:batch_index], y[i:batch_index], clip_name[i:batch_index]
                      batch_x, batch_y = self.toGPU(batch_x, torch.float32), self.toGPU(batch_y, torch.long)
                      #start = torch.cuda.Event(enable_timing=True)
                      #end =  torch.cuda.Event(enable_timing=True)
                      #start.record()
                      output = self.model.forward(batch_x).view(-1, 2)
                      #end.record()
                      #torch.cuda.synchronize()
                      inference_time += 0#start.elapsed_time(end)
                      loss_test = self.loss_func(output, batch_y)
                      acc_loss += loss_test.detach().cpu().item() * len(batch_y)
                      # store output, label statistics
                      self.update_categorical_outputs(categories, output, batch_y, batch_clip_name)
    
          # calculate one risk score per sequence (this is not implemented for each category)
          sum_seq_len = 0
          num_risky_sequences = 0
          num_safe_sequences = 0
          correct_risky_seq = 0
          correct_safe_seq = 0
          incorrect_risky_seq = 0
          incorrect_safe_seq = 0
          sequences = len(categories['lanechange']['labels'])
          for indices in range(sequences):
              seq_output = categories['lanechange']['outputs'][indices]
              label = categories['lanechange']['labels'][indices]
              pred = torch.argmax(seq_output)
              
              # risky clip
              if label == 1:
                  num_risky_sequences += 1
                  sum_seq_len += seq_output.shape[0]
                  correct_risky_seq += self.correctness(label, pred)
                  incorrect_risky_seq += self.correctness(label, pred)
              # non-risky clip
              elif label == 0:
                  num_safe_sequences += 1
                  incorrect_safe_seq += self.correctness(label, pred)
                  correct_safe_seq += self.correctness(label, pred)
          
          avg_risky_seq_len = sum_seq_len / num_risky_sequences # sequence length for comparison with the prediction frame metric. 
          seq_tpr = correct_risky_seq / num_risky_sequences
          seq_fpr = incorrect_safe_seq / num_safe_sequences
          seq_tnr = correct_safe_seq / num_safe_sequences
          seq_fnr = incorrect_risky_seq / num_risky_sequences
          if prof != None:
              prof_result = prof.key_averages().table(sort_by="cuda_time_total")
    
          return  categories, \
                  acc_loss/len(X), \
                  avg_risky_seq_len, \
                  inference_time, \
                  prof_result, \
                  seq_tpr, \
                  seq_fpr, \
                  seq_tnr, \
                  seq_fnr

    def correctness(self, output, pred):
        return 1 if output == pred else 0

    def update_categorical_outputs(self, categories, outputs, labels, clip_name):
        '''
            Aggregates output, label pairs for every driving category
        '''
        n = len(clip_name)
        for i in range(n):
            category = clip_name[i]
            # FIXME: probably better way to do this
            if category in categories:
                categories[category]['outputs'] = torch.cat([categories[category]['outputs'], torch.unsqueeze(outputs[i], dim=0)], dim=0)
                categories[category]['labels'] = torch.cat([categories[category]['labels'], torch.unsqueeze(labels[i], dim=0)], dim=0)
            # multi category
            else: 
                category = 'all'
                categories[category]['outputs'] = torch.cat([categories[category]['outputs'], torch.unsqueeze(outputs[i], dim=0)], dim=0)
                categories[category]['labels'] = torch.cat([categories[category]['labels'], torch.unsqueeze(labels[i], dim=0)], dim=0)

        # reshape outputs
        for k, v in categories.items():
            categories[k]['outputs'] = categories[k]['outputs'].reshape(-1, 2)
    
    def eval_model(self, current_epoch=None):
           metrics = {}
           categories_train, \
           acc_loss_train, \
           train_avg_seq_len, \
           train_inference_time, \
           train_profiler_result, \
           seq_tpr, seq_fpr, seq_tnr, seq_fnr = self.model_inference(self.training_data, self.training_labels, self.training_clip_name) 
    
           # Collect metrics from all driving categories
           for category in self.unique_clips.keys():
               if category == 'all':
                   metrics['train'] = get_metrics(categories_train['all']['outputs'], categories_train['all']['labels'])
                   metrics['train']['loss'] = acc_loss_train
                   metrics['train']['avg_seq_len'] = train_avg_seq_len
                   metrics['train']['seq_tpr'] = seq_tpr
                   metrics['train']['seq_tnr'] = seq_tnr
                   metrics['train']['seq_fpr'] = seq_fpr
                   metrics['train']['seq_fnr'] = seq_fnr
               elif category == 'lanechange':
                   metrics['train'] = get_metrics(categories_train['lanechange']['outputs'], categories_train['lanechange']['labels'])
                   metrics['train']['loss'] = acc_loss_train
                   metrics['train']['avg_seq_len'] = train_avg_seq_len
                   metrics['train']['seq_tpr'] = seq_tpr
                   metrics['train']['seq_tnr'] = seq_tnr
                   metrics['train']['seq_fpr'] = seq_fpr
                   metrics['train']['seq_fnr'] = seq_fnr
               else:
                   metrics['train'][category] = get_metrics(categories_train[category]['outputs'], categories_train[category]['labels'])
    
           categories_test, \
           acc_loss_test, \
           val_avg_seq_len, \
           test_inference_time, \
           test_profiler_result, \
           seq_tpr, seq_fpr, seq_tnr, seq_fnr = self.model_inference(self.testing_data, self.testing_labels, self.testing_clip_name) 
    
           # Collect metrics from all driving categories
           for category in self.unique_clips.keys():
               if category == 'all':
                   metrics['test'] = get_metrics(categories_test['all']['outputs'], categories_test['all']['labels'])
                   metrics['test']['loss'] = acc_loss_test
                   metrics['test']['avg_seq_len'] = val_avg_seq_len
                   metrics['test']['seq_tpr'] = seq_tpr
                   metrics['test']['seq_tnr'] = seq_tnr
                   metrics['test']['seq_fpr'] = seq_fpr
                   metrics['test']['seq_fnr'] = seq_fnr
                   metrics['avg_inf_time'] = (train_inference_time + test_inference_time) / ((len(self.training_labels) + len(self.testing_labels))*5)
               elif category == 'lanechange':
                   metrics['test'] = get_metrics(categories_test['lanechange']['outputs'], categories_test['lanechange']['labels'])
                   metrics['test']['loss'] = acc_loss_test
                   metrics['test']['avg_seq_len'] = val_avg_seq_len
                   metrics['test']['seq_tpr'] = seq_tpr
                   metrics['test']['seq_tnr'] = seq_tnr
                   metrics['test']['seq_fpr'] = seq_fpr
                   metrics['test']['seq_fnr'] = seq_fnr
                   metrics['avg_inf_time'] = (train_inference_time + test_inference_time) / ((len(self.training_labels) + len(self.testing_labels))*5)
               else:
                   metrics['test'][category] = get_metrics(categories_test[category]['outputs'], categories_test[category]['labels'])
    
           
           print("\ntrain loss: " + str(acc_loss_train) + ", acc:", metrics['train']['acc'], metrics['train']['confusion'], "mcc:", metrics['train']['mcc'], \
                 "\ntest loss: " +  str(acc_loss_test) + ", acc:",  metrics['test']['acc'],  metrics['test']['confusion'], "mcc:", metrics['test']['mcc'])
    
           self.update_im_best_metrics(metrics, current_epoch)
           metrics['best_epoch'] = self.best_epoch
           metrics['best_val_loss'] = self.best_val_loss
           metrics['best_val_acc'] = self.best_val_acc
           metrics['best_val_auc'] = self.best_val_auc
           metrics['best_val_conf'] = self.best_val_confusion
           metrics['best_val_f1'] = self.best_val_f1
           metrics['best_val_mcc'] = self.best_val_mcc
           metrics['best_val_acc_balanced'] = self.best_val_acc_balanced
           
           if self.config.training_configuration['n_fold'] <= 1 or self.log:  
               self.log2wandb(metrics)
    
           return categories_train, categories_test, metrics
   
    def update_im_best_metrics(self, metrics, current_epoch):
        if metrics['test']['loss'] < self.best_val_loss:
            self.best_val_loss = metrics['test']['loss']
            self.best_epoch = current_epoch if current_epoch != None else self.config.epochs
            self.best_val_acc = metrics['test']['acc']
            self.best_val_auc = metrics['test']['auc']
            self.best_val_confusion = metrics['test']['confusion']
            self.best_val_f1 = metrics['test']['f1']
            self.best_val_mcc = metrics['test']['mcc']
            self.best_val_acc_balanced = metrics['test']['balanced_acc']
            #self.save_model()
                    
    def update_im_cross_valid_metrics(self, categories_train, categories_test, metrics):
        '''
            Stores cross-validation metrics for all driving categories
        '''
        datasets = ['train', 'test']
        if self.fold == 1:
            for dataset in datasets:
                categories = categories_train if dataset == 'train' else categories_test
                for category in self.unique_clips.keys():
                    if category == 'all':
                        self.results['outputs'+'_'+dataset] = categories['all']['outputs']
                        self.results['labels'+'_'+dataset] = categories['all']['labels']
                        self.results[dataset] = metrics[dataset]
                        self.results[dataset]['loss'] = metrics[dataset]['loss']
                        self.results[dataset]['avg_seq_len']  = metrics[dataset]['avg_seq_len'] 
                        
                        # Best results
                        self.results['avg_inf_time']  = metrics['avg_inf_time']
                        self.results['best_epoch']    = metrics['best_epoch']
                        self.results['best_val_loss'] = metrics['best_val_loss']
                        self.results['best_val_acc']  = metrics['best_val_acc']
                        self.results['best_val_auc']  = metrics['best_val_auc']
                        self.results['best_val_conf'] = metrics['best_val_conf']
                        self.results['best_val_f1']   = metrics['best_val_f1']
                        self.results['best_val_mcc']  = metrics['best_val_mcc']
                        self.results['best_val_acc_balanced'] = metrics['best_val_acc_balanced']
                    else:
                        self.results[dataset][category]['outputs'] = categories[category]['outputs']
                        self.results[dataset][category]['labels'] = categories[category]['labels']
    
        else:
            for dataset in datasets:
                categories = categories_train if dataset == 'train' else categories_test
                for category in self.unique_clips.keys():
                    if category == 'all':
                        self.results['outputs'+'_'+dataset] = torch.cat((self.results['outputs'+'_'+dataset], categories['all']['outputs']), dim=0)
                        self.results['labels'+'_'+dataset]  = torch.cat((self.results['labels'+'_'+dataset], categories['all']['labels']), dim=0)
                        self.results[dataset]['loss'] = np.append(self.results[dataset]['loss'], metrics[dataset]['loss'])
                        self.results[dataset]['avg_seq_len']  = np.append(self.results[dataset]['avg_seq_len'], metrics[dataset]['avg_seq_len'])
                        
                        # Best results
                        self.results['avg_inf_time']  = np.append(self.results['avg_inf_time'], metrics['avg_inf_time'])
                        self.results['best_epoch']    = np.append(self.results['best_epoch'], metrics['best_epoch'])
                        self.results['best_val_loss'] = np.append(self.results['best_val_loss'], metrics['best_val_loss'])
                        self.results['best_val_acc']  = np.append(self.results['best_val_acc'], metrics['best_val_acc'])
                        self.results['best_val_auc']  = np.append(self.results['best_val_auc'], metrics['best_val_auc'])
                        self.results['best_val_conf'] = np.append(self.results['best_val_conf'], metrics['best_val_conf'])
                        self.results['best_val_f1']   = np.append(self.results['best_val_f1'], metrics['best_val_f1'])
                        self.results['best_val_mcc']  = np.append(self.results['best_val_mcc'], metrics['best_val_mcc'])
                        self.results['best_val_acc_balanced'] = np.append(self.results['best_val_acc_balanced'], metrics['best_val_acc_balanced'])
                    else:
                        self.results[dataset][category]['outputs'] = torch.cat((self.results[dataset][category]['outputs'], categories[category]['outputs']), dim=0)
                        self.results[dataset][category]['labels']  = torch.cat((self.results[dataset][category]['labels'], categories[category]['labels']), dim=0)
            
        # Log final averaged results
        if self.fold == self.config.training_configuration['n_fold']:
            final_metrics = {}
            for dataset in datasets:
                for category in self.unique_clips.keys():
                    if category == 'all':
                        final_metrics[dataset] = get_metrics(self.results['outputs'+'_'+dataset], self.results['labels'+'_'+dataset])
                        final_metrics[dataset]['loss'] = np.average(self.results[dataset]['loss'])
                        final_metrics[dataset]['avg_seq_len'] = np.average(self.results[dataset]['avg_seq_len'])
    
                        # Best results
                        final_metrics['avg_inf_time']  = np.average(self.results['avg_inf_time'])
                        final_metrics['best_epoch']    = np.average(self.results['best_epoch'])
                        final_metrics['best_val_loss'] = np.average(self.results['best_val_loss'])
                        final_metrics['best_val_acc']  = np.average(self.results['best_val_acc'])
                        final_metrics['best_val_auc']  = np.average(self.results['best_val_auc'])
                        final_metrics['best_val_conf'] = self.results['best_val_conf']
                        final_metrics['best_val_f1']   = np.average(self.results['best_val_f1'])
                        final_metrics['best_val_mcc']  = np.average(self.results['best_val_mcc'])
                        final_metrics['best_val_acc_balanced'] = np.average(self.results['best_val_acc_balanced'])
                    else: 
                        final_metrics[dataset][category] = get_metrics(self.results[dataset][category]['outputs'], self.results[dataset][category]['labels'])
    
            print('\nFinal Averaged Results')
            print("\naverage train loss: " + str(final_metrics['train']['loss']) + ", average acc:", final_metrics['train']['acc'], final_metrics['train']['confusion'], final_metrics['train']['auc'], \
                "\naverage test loss: " +  str(final_metrics['test']['loss']) + ", average acc:", final_metrics['test']['acc'],  final_metrics['test']['confusion'], final_metrics['test']['auc'])
    
            self.log2wandb(final_metrics)
            
            # final combined results and metrics
            return self.results['outputs_train'], self.results['labels_train'], self.results['outputs_test'], self.results['labels_test'], final_metrics

    def cross_valid(self):
        # KFold cross validation with similar class distribution in each fold
        skf = StratifiedKFold(n_splits=self.config.training_configuration["n_fold"])
        X = np.append(self.training_data, self.testing_data, axis=0)
        clip_name = np.append(self.training_clip_name, self.testing_clip_name, axis=0)
        y = np.append(self.training_labels, self.testing_labels, axis=0)

        # self.results stores average metrics for the the n_folds
        self.results = {}
        self.fold = 1

        # Split training and testing data based on n_splits (Folds)
        for train_index, test_index in skf.split(X, y):
            self.training_data, self.testing_data, self.training_clip_name, self.testing_clip_name, self.training_labels, self.testing_labels = None, None, None, None, None, None #clear vars to save memory
            X_train, X_test = X[train_index], X[test_index]
            clip_train, clip_test = clip_name[train_index], clip_name[test_index]
            y_train, y_test = y[train_index], y[test_index]
            self.class_weights = torch.from_numpy(compute_class_weight('balanced', np.unique(y_train), y_train))

            # Update dataset
            self.training_data = X_train
            self.testing_data  = X_test
            self.training_clip_name = clip_train
            self.testing_clip_name = clip_test
            self.training_labels = y_train
            self.testing_labels  = y_test

            print('\nFold {}'.format(self.fold))
            print("Number of Training Sequences Included: ", len(X_train))
            print("Number of Testing Sequences Included: ",  len(X_test))
            print("Num of Training Labels in Each Class: " + str(np.unique(self.training_labels, return_counts=True)[1]) + ", Class Weights: " + str(self.class_weights))
            print("Num of Testing Labels in Each Class: "  + str(np.unique(self.testing_labels, return_counts=True)[1]) + ", Class Weights: " + str(self.class_weights))
           
            self.best_val_loss = 99999
            self.train()
            self.log = True
            categories_train, categories_test, metrics = self.eval_model(self.fold)
            self.update_cross_valid_metrics(categories_train, categories_test, metrics)
            self.log = False

            if self.fold != self.config.training_configuration["n_fold"]:
                self.reset_weights(self.model)
                del self.optimizer
                self.build_model(self.model)
                
            self.fold += 1            
        del self.results


    def update_cross_valid_metrics(self, categories_train, categories_test, metrics):
            '''
                Stores cross-validation metrics for all driving categories
            '''
            datasets = ['train', 'test']
            if self.fold == 1:
                for dataset in datasets:
                    categories = categories_train if dataset == 'train' else categories_test
                    for category in self.unique_clips.keys():
                        if category == 'all':
                            self.results['outputs'+'_'+dataset] = categories['all']['outputs']
                            self.results['labels'+'_'+dataset] = categories['all']['labels']
                            self.results[dataset] = metrics[dataset]
                            self.results[dataset]['loss'] = metrics[dataset]['loss']
                            self.results[dataset]['avg_seq_len']  = metrics[dataset]['avg_seq_len'] 
                            
                            # Best results
                            self.results['avg_inf_time']  = metrics['avg_inf_time']
                            self.results['best_epoch']    = metrics['best_epoch']
                            self.results['best_val_loss'] = metrics['best_val_loss']
                            self.results['best_val_acc']  = metrics['best_val_acc']
                            self.results['best_val_auc']  = metrics['best_val_auc']
                            self.results['best_val_conf'] = metrics['best_val_conf']
                            self.results['best_val_f1']   = metrics['best_val_f1']
                            self.results['best_val_mcc']  = metrics['best_val_mcc']
                            self.results['best_val_acc_balanced'] = metrics['best_val_acc_balanced']
                        elif category == "lanechange":
                            self.results['outputs'+'_'+dataset] = categories['all']['outputs']
                            self.results['labels'+'_'+dataset] = categories['all']['labels']
                            self.results[dataset] = metrics[dataset]
                            self.results[dataset]['loss'] = metrics[dataset]['loss']
                            self.results[dataset]['avg_seq_len']  = metrics[dataset]['avg_seq_len'] 
                            
                            # Best results
                            self.results['avg_inf_time']  = metrics['avg_inf_time']
                            self.results['best_epoch']    = metrics['best_epoch']
                            self.results['best_val_loss'] = metrics['best_val_loss']
                            self.results['best_val_acc']  = metrics['best_val_acc']
                            self.results['best_val_auc']  = metrics['best_val_auc']
                            self.results['best_val_conf'] = metrics['best_val_conf']
                            self.results['best_val_f1']   = metrics['best_val_f1']
                            self.results['best_val_mcc']  = metrics['best_val_mcc']
                            self.results['best_val_acc_balanced'] = metrics['best_val_acc_balanced']                            
                        else:
                            self.results[dataset][category]['outputs'] = categories[category]['outputs']
                            self.results[dataset][category]['labels'] = categories[category]['labels']
    
            else:
                for dataset in datasets:
                    categories = categories_train if dataset == 'train' else categories_test
                    for category in self.unique_clips.keys():
                        if category == 'all':
                            self.results['outputs'+'_'+dataset] = torch.cat((self.results['outputs'+'_'+dataset], categories['all']['outputs']), dim=0)
                            self.results['labels'+'_'+dataset]  = torch.cat((self.results['labels'+'_'+dataset], categories['all']['labels']), dim=0)
                            self.results[dataset]['loss'] = np.append(self.results[dataset]['loss'], metrics[dataset]['loss'])
                            self.results[dataset]['avg_seq_len']  = np.append(self.results[dataset]['avg_seq_len'], metrics[dataset]['avg_seq_len'])
                            
                            # Best results
                            self.results['avg_inf_time']  = np.append(self.results['avg_inf_time'], metrics['avg_inf_time'])
                            self.results['best_epoch']    = np.append(self.results['best_epoch'], metrics['best_epoch'])
                            self.results['best_val_loss'] = np.append(self.results['best_val_loss'], metrics['best_val_loss'])
                            self.results['best_val_acc']  = np.append(self.results['best_val_acc'], metrics['best_val_acc'])
                            self.results['best_val_auc']  = np.append(self.results['best_val_auc'], metrics['best_val_auc'])
                            self.results['best_val_conf'] = np.append(self.results['best_val_conf'], metrics['best_val_conf'])
                            self.results['best_val_f1']   = np.append(self.results['best_val_f1'], metrics['best_val_f1'])
                            self.results['best_val_mcc']  = np.append(self.results['best_val_mcc'], metrics['best_val_mcc'])
                            self.results['best_val_acc_balanced'] = np.append(self.results['best_val_acc_balanced'], metrics['best_val_acc_balanced'])
                        elif category == "lanechange":
                            self.results['outputs'+'_'+dataset] = torch.cat((self.results['outputs'+'_'+dataset], categories['all']['outputs']), dim=0)
                            self.results['labels'+'_'+dataset]  = torch.cat((self.results['labels'+'_'+dataset], categories['all']['labels']), dim=0)
                            self.results[dataset]['loss'] = np.append(self.results[dataset]['loss'], metrics[dataset]['loss'])
                            self.results[dataset]['avg_seq_len']  = np.append(self.results[dataset]['avg_seq_len'], metrics[dataset]['avg_seq_len'])
                            
                            # Best results
                            self.results['avg_inf_time']  = np.append(self.results['avg_inf_time'], metrics['avg_inf_time'])
                            self.results['best_epoch']    = np.append(self.results['best_epoch'], metrics['best_epoch'])
                            self.results['best_val_loss'] = np.append(self.results['best_val_loss'], metrics['best_val_loss'])
                            self.results['best_val_acc']  = np.append(self.results['best_val_acc'], metrics['best_val_acc'])
                            self.results['best_val_auc']  = np.append(self.results['best_val_auc'], metrics['best_val_auc'])
                            self.results['best_val_conf'] = np.append(self.results['best_val_conf'], metrics['best_val_conf'])
                            self.results['best_val_f1']   = np.append(self.results['best_val_f1'], metrics['best_val_f1'])
                            self.results['best_val_mcc']  = np.append(self.results['best_val_mcc'], metrics['best_val_mcc'])
                            self.results['best_val_acc_balanced'] = np.append(self.results['best_val_acc_balanced'], metrics['best_val_acc_balanced'])
                        else:
                            self.results[dataset][category]['outputs'] = torch.cat((self.results[dataset][category]['outputs'], categories[category]['outputs']), dim=0)
                            self.results[dataset][category]['labels']  = torch.cat((self.results[dataset][category]['labels'], categories[category]['labels']), dim=0)
                
            # Log final averaged results
            if self.fold ==  self.config.training_configuration["n_fold"]:
                final_metrics = {}
                for dataset in datasets:
                    for category in self.unique_clips.keys():
                        if category == 'all':
                            final_metrics[dataset] = get_metrics(self.results['outputs'+'_'+dataset], self.results['labels'+'_'+dataset])
                            final_metrics[dataset]['loss'] = np.average(self.results[dataset]['loss'])
                            final_metrics[dataset]['avg_seq_len'] = np.average(self.results[dataset]['avg_seq_len'])
    
                            # Best results
                            final_metrics['avg_inf_time']  = np.average(self.results['avg_inf_time'])
                            final_metrics['best_epoch']    = np.average(self.results['best_epoch'])
                            final_metrics['best_val_loss'] = np.average(self.results['best_val_loss'])
                            final_metrics['best_val_acc']  = np.average(self.results['best_val_acc'])
                            final_metrics['best_val_auc']  = np.average(self.results['best_val_auc'])
                            final_metrics['best_val_conf'] = self.results['best_val_conf']
                            final_metrics['best_val_f1']   = np.average(self.results['best_val_f1'])
                            final_metrics['best_val_mcc']  = np.average(self.results['best_val_mcc'])
                            final_metrics['best_val_acc_balanced'] = np.average(self.results['best_val_acc_balanced'])
                        elif category == 'lanechange':
                            final_metrics[dataset] = get_metrics(self.results['outputs'+'_'+dataset], self.results['labels'+'_'+dataset])
                            final_metrics[dataset]['loss'] = np.average(self.results[dataset]['loss'])
                            final_metrics[dataset]['avg_seq_len'] = np.average(self.results[dataset]['avg_seq_len'])
    
                            # Best results
                            final_metrics['avg_inf_time']  = np.average(self.results['avg_inf_time'])
                            final_metrics['best_epoch']    = np.average(self.results['best_epoch'])
                            final_metrics['best_val_loss'] = np.average(self.results['best_val_loss'])
                            final_metrics['best_val_acc']  = np.average(self.results['best_val_acc'])
                            final_metrics['best_val_auc']  = np.average(self.results['best_val_auc'])
                            final_metrics['best_val_conf'] = self.results['best_val_conf']
                            final_metrics['best_val_f1']   = np.average(self.results['best_val_f1'])
                            final_metrics['best_val_mcc']  = np.average(self.results['best_val_mcc'])
                            final_metrics['best_val_acc_balanced'] = np.average(self.results['best_val_acc_balanced'])
                        else: 
                            final_metrics[dataset][category] = get_metrics(self.results[dataset][category]['outputs'], self.results[dataset][category]['labels'])
    
                print('\nFinal Averaged Results')
                print("\naverage train loss: " + str(final_metrics['train']['loss']) + ", average acc:", final_metrics['train']['acc'], final_metrics['train']['confusion'], final_metrics['train']['auc'], \
                    "\naverage test loss: " +  str(final_metrics['test']['loss']) + ", average acc:", final_metrics['test']['acc'],  final_metrics['test']['confusion'], final_metrics['test']['auc'])
    
                self.log2wandb(final_metrics)
                
                # final combined results and metrics
                return self.results['outputs_train'], self.results['labels_train'], self.results['outputs_test'], self.results['labels_test'], final_metrics


        
    def log2wandb(self, metrics):
        '''
            Log metrics from all driving categories
        '''
        for category in self.unique_clips.keys():
            if category == 'all':
                log_im_wandb(metrics)
            elif category == 'lanechange':
                log_im_wandb(metrics)
            else:
                log_wandb_categories(metrics, id=category)